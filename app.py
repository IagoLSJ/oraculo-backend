import os
import uuid
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from flask import Flask, request, jsonify
from flask_cors import CORS
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from sktime.forecasting.arima import AutoARIMA
import traceback
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Backend não-interativo para evitar problemas

# --- Configuração de Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuração da Aplicação ---
app = Flask(__name__)
CORS(app)


# Configurações como constantes
class Config:
    UPLOAD_FOLDER = Path("uploads")
    IMAGES_FOLDER = Path("public_imagens")  # NOVA PASTA PARA IMAGENS
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'.csv'}
    MIN_DATA_POINTS = 4


Config.UPLOAD_FOLDER.mkdir(exist_ok=True)
Config.IMAGES_FOLDER.mkdir(exist_ok=True)  # CRIAR PASTA DE IMAGENS
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH


# --- Exceções Customizadas ---
class DataProcessingError(Exception):
    """Exceção para erros de processamento de dados"""
    pass


class InsufficientDataError(Exception):
    """Exceção para dados insuficientes"""
    pass


# --- Classes de Dados ---
@dataclass
class AnalysisParams:
    selected_unidades: Optional[List[str]] = None
    selected_semestre: Optional[str] = None


@dataclass
class FileDetails:
    unidades: List[str]
    semestres: List[str]
    preview_data: Dict[str, Any]


# --- Validadores ---
class DataValidator:
    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        return Path(filename).suffix.lower() in Config.ALLOWED_EXTENSIONS

    @staticmethod
    def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
        return all(col in df.columns for col in required_columns)

    @staticmethod
    def validate_data_sufficiency(series: pd.Series) -> bool:
        return len(series.dropna()) >= Config.MIN_DATA_POINTS


# --- Utilitários ---
class DataUtils:
    @staticmethod
    def parse_semester_to_timestamp(value: Union[str, float]) -> Optional[pd.Timestamp]:
        if isinstance(value, str) and '-' in value:
            try:
                return pd.to_datetime(value)
            except ValueError:
                return None
        try:
            year, semester = map(int, str(value).split('.'))
            month = 1 if semester == 1 else 7
            return pd.Timestamp(year=year, month=month, day=1)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Erro ao converter semestre {value}: {e}")
            return None

    @staticmethod
    def clean_percentage_column(series: pd.Series) -> pd.Series:
        if series.dtype == 'object':
            series = series.astype(str).str.replace('%', '').str.replace(',', '.')
        return pd.to_numeric(series, errors='coerce')

    @staticmethod
    def safe_json_serialize(obj: Any) -> Any:
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        return obj


# --- Classe Principal de Análise ---
class EvasionAnalyzer:
    def __init__(self):
        self.column_mapping = {
            'Unidade': 'unidade_academica',
            'Semestre': 'semestre',
            'Taxa de Evasão': 'taxa_evasao',
            'Unidade Academica': 'unidade_academica'
        }
        self.required_columns = ['unidade_academica', 'semestre', 'taxa_evasao']

    def load_and_clean_data(self, file_path: Optional[str] = None,
                            json_data: Optional[Dict] = None) -> pd.DataFrame:
        try:
            if file_path:
                df = pd.read_csv(file_path)
            elif json_data:
                df = pd.DataFrame(json_data['rows'], columns=json_data['headers'])
            else:
                raise ValueError("É necessário fornecer file_path ou json_data")
            df = self._clean_and_transform_data(df)
            return df
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            raise DataProcessingError(f"Erro ao processar dados: {str(e)}")

    def _clean_and_transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=self.column_mapping)
        if not DataValidator.validate_required_columns(df, self.required_columns):
            missing_cols = set(self.required_columns) - set(df.columns)
            raise DataProcessingError(f"Colunas obrigatórias ausentes: {missing_cols}.")
        df['semestre'] = df['semestre'].apply(DataUtils.parse_semester_to_timestamp)
        df['taxa_evasao'] = DataUtils.clean_percentage_column(df['taxa_evasao'])
        df = df.dropna(subset=['semestre', 'taxa_evasao'])
        df = df.set_index('semestre').sort_index()
        return df

    # --- MÉTODOS PARA SALVAR GRÁFICOS EM PASTA ---
    def _save_plot_to_file(self, fig, filename: str) -> str:
        """Salva uma figura matplotlib como arquivo PNG e retorna o caminho."""
        filepath = Config.IMAGES_FOLDER / f"{filename}.png"
        fig.savefig(str(filepath), format='png', bbox_inches='tight', dpi=300)
        plt.close(fig)
        logger.info(f"Gráfico salvo em: {filepath}")
        return str(filepath)

    def _generate_unique_filename(self, base_name: str, analysis_id: str) -> str:
        """Gera um nome único para o arquivo baseado no ID da análise."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{analysis_id}_{base_name}_{timestamp}"

    def _generate_decomposition_plot(self, series: pd.Series, decomposition_result: Dict[str, Any], analysis_id: str) -> \
    Optional[str]:
        """Gera o gráfico de decomposição e salva na pasta."""
        try:
            fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            fig.suptitle('Decomposição da Série Temporal', fontsize=16)

            # Série Original
            axs[0].plot(series.index, series.values, label='Original', color='blue', linewidth=2)
            axs[0].set_ylabel('Taxa (%)')
            axs[0].legend()
            axs[0].grid(True, alpha=0.3)

            # Tendência
            if decomposition_result and 'trend' in decomposition_result:
                trend_x = pd.to_datetime(decomposition_result['trend']['x'])
                axs[1].plot(trend_x, decomposition_result['trend']['y'],
                            label='Tendência', color='orange', linewidth=2)
                axs[1].set_ylabel('Tendência')
                axs[1].legend()
                axs[1].grid(True, alpha=0.3)

            # Sazonalidade
            if decomposition_result and 'seasonal' in decomposition_result:
                seasonal_x = pd.to_datetime(decomposition_result['seasonal']['x'])
                axs[2].plot(seasonal_x, decomposition_result['seasonal']['y'],
                            label='Sazonalidade', color='green', linewidth=2)
                axs[2].set_ylabel('Sazonalidade')
                axs[2].legend()
                axs[2].grid(True, alpha=0.3)

            # Resíduos
            if decomposition_result and 'residual' in decomposition_result:
                residual_x = pd.to_datetime(decomposition_result['residual']['x'])
                axs[3].scatter(residual_x, decomposition_result['residual']['y'],
                               label='Resíduos', color='red', alpha=0.7)
                axs[3].axhline(0, color='grey', linestyle='--', alpha=0.8)
                axs[3].set_ylabel('Resíduos')
                axs[3].set_xlabel('Semestre')
                axs[3].legend()
                axs[3].grid(True, alpha=0.3)

            plt.xticks(rotation=45)
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            filename = self._generate_unique_filename("decomposicao", analysis_id)
            return self._save_plot_to_file(fig, filename)

        except Exception as e:
            logger.warning(f"Erro ao gerar gráfico de decomposição: {e}")
            return None

    def _generate_forecast_plot(self, train_series: pd.Series, forecast_result: Dict[str, Any], analysis_id: str) -> \
    Optional[str]:
        """Gera o gráfico de previsão e salva na pasta."""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_title('Previsão da Taxa de Evasão', fontsize=16)

            # Dados de treino
            ax.plot(train_series.index, train_series.values,
                    label='Treino', marker='o', linewidth=2, color='blue')

            # Dados de teste (se disponível)
            if forecast_result.get('test_x') and forecast_result.get('test_y'):
                test_x = pd.to_datetime(forecast_result['test_x'])
                ax.plot(test_x, forecast_result['test_y'],
                        label='Teste', marker='o', linewidth=2, color='green')

            # Previsões
            forecast_x = pd.to_datetime(forecast_result['forecast_x'])
            ax.plot(forecast_x, forecast_result['forecast_y'],
                    label='Previsão', marker='s', linestyle='--', linewidth=2, color='red')

            # Intervalo de confiança
            if forecast_result.get('forecast_ci_lower') and forecast_result.get('forecast_ci_upper'):
                lower = forecast_result['forecast_ci_lower']
                upper = forecast_result['forecast_ci_upper']
                ax.fill_between(forecast_x, lower, upper,
                                color='red', alpha=0.2, label='IC 95%')

            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Semestre', fontsize=12)
            ax.set_ylabel('Taxa de Evasão (%)', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()

            filename = self._generate_unique_filename("previsao", analysis_id)
            return self._save_plot_to_file(fig, filename)

        except Exception as e:
            logger.warning(f"Erro ao gerar gráfico de previsão: {e}")
            return None

    def _generate_autocorrelation_plot(self, series: pd.Series, autocorr_result: Dict[str, Any], analysis_id: str) -> \
    Optional[Dict[str, str]]:
        """Gera os gráficos de ACF e PACF e salva na pasta."""
        try:
            if not autocorr_result:
                return None

            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

            # Gerar ACF
            fig_acf = plt.figure(figsize=(10, 5))
            plot_acf(series, ax=fig_acf.gca(), lags=min(10, len(series) // 2 - 1))
            fig_acf.gca().set_title('Função de Autocorrelação (ACF)', fontsize=14)
            fig_acf.gca().grid(True, alpha=0.3)
            acf_filename = self._generate_unique_filename("acf", analysis_id)
            acf_path = self._save_plot_to_file(fig_acf, acf_filename)

            # Gerar PACF
            fig_pacf = plt.figure(figsize=(10, 5))
            plot_pacf(series, ax=fig_pacf.gca(), lags=min(10, len(series) // 2 - 1))
            fig_pacf.gca().set_title('Função de Autocorrelação Parcial (PACF)', fontsize=14)
            fig_pacf.gca().grid(True, alpha=0.3)
            pacf_filename = self._generate_unique_filename("pacf", analysis_id)
            pacf_path = self._save_plot_to_file(fig_pacf, pacf_filename)

            return {'acf_plot': acf_path, 'pacf_plot': pacf_path}

        except Exception as e:
            logger.warning(f"Erro ao gerar gráfico de autocorrelação: {e}")
            return None

    # --- MÉTODOS DE ANÁLISE PRINCIPAL ---
    def perform_analysis(self, data_series: pd.Series, semestre_corte_str: Optional[str] = None,
                         analysis_id: Optional[str] = None) -> Dict[str, Any]:
        """Executa análise completa, separando treino e teste."""
        try:
            # Gerar ID único para esta análise se não fornecido
            if analysis_id is None:
                analysis_id = str(uuid.uuid4())[:8]

            clean_series = data_series.dropna()

            if not DataValidator.validate_data_sufficiency(clean_series):
                raise InsufficientDataError(
                    f"Dados insuficientes. Mínimo: {Config.MIN_DATA_POINTS}, atual: {len(clean_series)}")

            train_series = clean_series
            test_series = None
            if semestre_corte_str:
                semestre_corte_ts = DataUtils.parse_semester_to_timestamp(semestre_corte_str)
                if semestre_corte_ts and semestre_corte_ts in clean_series.index:
                    train_series = clean_series[clean_series.index <= semestre_corte_ts]
                    test_series = clean_series[clean_series.index > semestre_corte_ts]

            if not DataValidator.validate_data_sufficiency(train_series):
                raise InsufficientDataError("Dados de treino insuficientes após o corte.")

            # Geração dos dados para os gráficos interativos
            results = {
                'analysis_id': analysis_id,
                'decomposition': self._perform_decomposition(train_series),
                'forecast': self._perform_forecast(train_series, test_series),
                'statistics': self._calculate_statistics(train_series),
                'autocorrelation': self._perform_autocorrelation(train_series)
            }

            # MODIFICAÇÃO: Gerar e salvar os gráficos na pasta public_imagens
            plot_paths = {
                'decomposition_plot': self._generate_decomposition_plot(train_series, results['decomposition'],
                                                                        analysis_id),
                'forecast_plot': self._generate_forecast_plot(train_series, results['forecast'], analysis_id),
                'autocorrelation_plots': self._generate_autocorrelation_plot(train_series, results['autocorrelation'],
                                                                             analysis_id)
            }

            # Adicionar caminhos dos arquivos ao resultado
            results['plot_files'] = plot_paths

            logger.info(f"Análise {analysis_id} concluída com sucesso. Gráficos salvos em: {Config.IMAGES_FOLDER}")
            return results
        except InsufficientDataError:
            raise
        except Exception as e:
            logger.error(f"Erro na análise: {e}")
            raise DataProcessingError(f"Erro durante análise: {str(e)}")

    def _perform_decomposition(self, series: pd.Series) -> Optional[Dict[str, Any]]:
        try:
            period = 2 if len(series) < 4 else max(2, len(series) // 2)
            decomposition = seasonal_decompose(series, model='additive', period=period)
            return {
                "original": {"x": [DataUtils.safe_json_serialize(x) for x in series.index],
                             "y": series.values.tolist()},
                "trend": {"x": [DataUtils.safe_json_serialize(x) for x in decomposition.trend.dropna().index],
                          "y": decomposition.trend.dropna().values.tolist()},
                "seasonal": {"x": [DataUtils.safe_json_serialize(x) for x in decomposition.seasonal.index],
                             "y": decomposition.seasonal.values.tolist()},
                "residual": {"x": [DataUtils.safe_json_serialize(x) for x in decomposition.resid.dropna().index],
                             "y": decomposition.resid.dropna().values.tolist()}
            }
        except Exception as e:
            logger.warning(f"Erro na decomposição: {e}")
            return None

    def _perform_forecast(self, train_series: pd.Series, test_series: Optional[pd.Series] = None) -> Optional[
        Dict[str, Any]]:
        try:
            numeric_train_series = pd.Series(train_series.values, index=pd.RangeIndex(len(train_series)))
            model = AutoARIMA(seasonal=False, stepwise=True, suppress_warnings=True, max_p=3, max_q=3, max_d=2)
            model.fit(numeric_train_series)
            results = {
                "original_x": [DataUtils.safe_json_serialize(x) for x in train_series.index],
                "original_y": train_series.values.tolist(), "test_x": [], "test_y": [],
                "forecast_x": [], "forecast_y": [], "forecast_ci_lower": [], "forecast_ci_upper": [],
                "mape": None
            }

            if test_series is not None and not test_series.empty:
                n_test_points = len(test_series)
                fh = list(range(1, n_test_points + 1))

                # Obter previsões e intervalos separadamente
                predictions = model.predict(fh=fh)
                pred_interval = model.predict_interval(fh=fh, coverage=0.95)

                # Verificar estrutura do pred_interval
                logger.info(f"Pred_interval shape: {pred_interval.shape}, columns: {pred_interval.columns.tolist()}")

                mape = mean_absolute_percentage_error(test_series.values, predictions.values) * 100

                # Extrair intervalos de confiança de forma segura
                if pred_interval.shape[1] >= 2:
                    ci_lower = pred_interval.iloc[:, 0].values.tolist()  # Limite inferior
                    ci_upper = pred_interval.iloc[:, 1].values.tolist()  # Limite superior
                else:
                    ci_lower = predictions.values.tolist()  # Fallback para predições
                    ci_upper = predictions.values.tolist()

                results.update({
                    "test_x": [DataUtils.safe_json_serialize(x) for x in test_series.index],
                    "test_y": test_series.values.tolist(),
                    "forecast_x": [DataUtils.safe_json_serialize(x) for x in test_series.index],
                    "forecast_y": predictions.values.tolist(),
                    "forecast_ci_lower": ci_lower,
                    "forecast_ci_upper": ci_upper,
                    "mape": float(mape)
                })
            else:
                fh = [1, 2, 3, 4]

                # Obter previsões e intervalos separadamente
                predictions = model.predict(fh=fh)
                pred_interval = model.predict_interval(fh=fh, coverage=0.95)

                # Verificar estrutura do pred_interval
                logger.info(f"Pred_interval shape: {pred_interval.shape}, columns: {pred_interval.columns.tolist()}")

                last_date = train_series.index[-1]
                forecast_dates = []
                current_date = last_date
                for _ in range(4):
                    if current_date.month == 1:
                        next_date = current_date.replace(month=7)
                    else:
                        next_date = current_date.replace(year=current_date.year + 1, month=1)
                    forecast_dates.append(next_date)
                    current_date = next_date

                # Extrair intervalos de confiança de forma segura
                if pred_interval.shape[1] >= 2:
                    ci_lower = pred_interval.iloc[:, 0].values.tolist()  # Limite inferior
                    ci_upper = pred_interval.iloc[:, 1].values.tolist()  # Limite superior
                else:
                    ci_lower = predictions.values.tolist()  # Fallback para predições
                    ci_upper = predictions.values.tolist()

                results.update({
                    "forecast_x": [DataUtils.safe_json_serialize(x) for x in forecast_dates],
                    "forecast_y": predictions.values.tolist(),
                    "forecast_ci_lower": ci_lower,
                    "forecast_ci_upper": ci_upper,
                })
            return results
        except Exception as e:
            logger.error(f"Erro na previsão: {e}\n{traceback.format_exc()}")
            return None

    def _calculate_statistics(self, series: pd.Series) -> Dict[str, float]:
        return {
            "mean": float(series.mean()), "std": float(series.std()),
            "min": float(series.min()), "max": float(series.max()),
            "trend": float(series.iloc[-1] - series.iloc[0]) if len(series) > 1 else 0.0
        }

    def _perform_autocorrelation(self, series: pd.Series) -> Optional[Dict[str, Any]]:
        try:
            nlags = min(10, len(series) // 2 - 1)
            if nlags <= 0: return None
            acf_values, acf_confint = acf(series, nlags=nlags, alpha=0.05)
            pacf_values, pacf_confint = pacf(series, nlags=nlags, alpha=0.05)
            return {
                "lags": list(range(len(acf_values))),
                "acf": acf_values.tolist(), "pacf": pacf_values.tolist(),
                "acf_confidence_upper": (acf_confint[:, 1] - acf_values).tolist(),
                "pacf_confidence_upper": (pacf_confint[:, 1] - pacf_values).tolist(),
            }
        except Exception as e:
            logger.warning(f"Erro no cálculo de autocorrelação: {e}")
            return None


# --- Handlers de Erro e Endpoints ---
@app.errorhandler(413)
def file_too_large(error): return jsonify({"error": "Arquivo muito grande. Máximo: 16MB"}), 413


@app.errorhandler(DataProcessingError)
def handle_data_processing_error(error): return jsonify({"error": str(error)}), 400


@app.errorhandler(InsufficientDataError)
def handle_insufficient_data_error(error): return jsonify({"error": str(error)}), 400


@app.route("/api/upload", methods=["POST"])
def upload_file():
    try:
        if 'file' not in request.files: return jsonify({"error": "Nenhum arquivo enviado"}), 400
        file = request.files['file']
        if not file.filename or not DataValidator.validate_file_extension(file.filename):
            return jsonify({"error": "Arquivo inválido ou não é CSV"}), 400
        file_id = str(uuid.uuid4())
        filepath = Config.UPLOAD_FOLDER / f"{file_id}.csv"
        file.save(str(filepath))
        return jsonify({"fileId": file_id, "fileName": file.filename}), 201
    except Exception as e:
        logger.error(f"Erro no upload: {e}")
        return jsonify({"error": "Erro interno do servidor"}), 500


@app.route("/api/files/<file_id>/details", methods=["GET"])
def get_file_details(file_id: str):
    try:
        filepath = Config.UPLOAD_FOLDER / f"{file_id}.csv"
        if not filepath.exists(): return jsonify({"error": "Arquivo não encontrado"}), 404
        analyzer = EvasionAnalyzer()
        df = analyzer.load_and_clean_data(file_path=str(filepath))
        unidades = sorted(df['unidade_academica'].unique().tolist()) if 'unidade_academica' in df.columns else []
        semestres = sorted(
            df.index.to_series().apply(lambda x: f"{x.year}.{1 if x.month == 1 else 2}").unique().tolist())
        preview_df = df.reset_index()
        preview_data = {"headers": preview_df.columns.tolist(), "rows": preview_df.astype(str).values.tolist()}
        file_details = FileDetails(unidades=unidades, semestres=semestres, preview_data=preview_data)
        return jsonify(file_details.__dict__), 200
    except DataProcessingError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Erro ao obter detalhes: {e}")
        return jsonify({"error": "Erro interno do servidor"}), 500


@app.route("/api/analysis", methods=["POST"])
def perform_analysis_route():
    try:
        payload = request.get_json()
        if not payload: return jsonify({"error": "Payload JSON inválido"}), 400
        params = AnalysisParams(
            selected_unidades=payload.get('params', {}).get('selectedUnidades'),
            selected_semestre=payload.get('params', {}).get('selectedSemestre')
        )
        edited_data = payload.get('data')
        if not edited_data: return jsonify({"error": "Dados para análise não fornecidos"}), 400

        # Gerar ID único para esta análise
        analysis_id = str(uuid.uuid4())[:8]

        analyzer = EvasionAnalyzer()
        df = analyzer.load_and_clean_data(json_data=edited_data)

        df_filtered = df
        if 'unidade_academica' in df.columns and params.selected_unidades:
            df_filtered = df[df['unidade_academica'].isin(params.selected_unidades)]
        time_series = df_filtered.groupby(df_filtered.index)['taxa_evasao'].mean()
        analysis_results = analyzer.perform_analysis(
            data_series=time_series,
            semestre_corte_str=params.selected_semestre,
            analysis_id=analysis_id
        )
        return jsonify(analysis_results), 200
    except (DataProcessingError, InsufficientDataError) as e:
        raise e
    except Exception as e:
        logger.error(f"Erro na análise: {e}")
        traceback.print_exc()
        return jsonify({"error": "Erro interno do servidor"}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "2.4"})


# --- Execução ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)