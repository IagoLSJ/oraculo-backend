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
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'.csv'}
    MIN_DATA_POINTS = 4


Config.UPLOAD_FOLDER.mkdir(exist_ok=True)
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

    def perform_analysis(self, data_series: pd.Series, semestre_corte_str: Optional[str] = None) -> Dict[str, Any]:
        """Executa análise completa, separando treino e teste."""
        try:
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

            results = {
                'decomposition': self._perform_decomposition(train_series),
                'forecast': self._perform_forecast(train_series, test_series),
                'statistics': self._calculate_statistics(train_series),
                'autocorrelation': self._perform_autocorrelation(train_series)
            }
            logger.info("Análise concluída com sucesso")
            return results
        except InsufficientDataError:
            raise
        except Exception as e:
            logger.error(f"Erro na análise: {e}")
            raise DataProcessingError(f"Erro durante análise: {str(e)}")

    def _perform_decomposition(self, series: pd.Series) -> Optional[Dict[str, Any]]:
        try:
            # Garante que o período seja no mínimo 2 e no máximo metade do tamanho da série
            period = 2 if len(series) < 4 else max(2, len(series) // 2)
            decomposition = seasonal_decompose(series, model='additive', period=period)

            # AJUSTE: Incluindo a série original na resposta para o frontend
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
            # O AutoARIMA da sktime funciona melhor com um índice numérico simples
            numeric_train_series = pd.Series(train_series.values, index=pd.RangeIndex(len(train_series)))

            model = AutoARIMA(seasonal=False, stepwise=True, suppress_warnings=True, max_p=3, max_q=3, max_d=2)
            model.fit(numeric_train_series)

            results = {
                "original_x": [DataUtils.safe_json_serialize(x) for x in train_series.index],
                "original_y": train_series.values.tolist(),
                "test_x": [], "test_y": [],
                "forecast_x": [], "forecast_y": [],
                "forecast_ci_lower": [], "forecast_ci_upper": [],  # AJUSTE: Adicionado para o frontend
                "mape": None
            }

            if test_series is not None and not test_series.empty:
                n_test_points = len(test_series)
                fh = list(range(1, n_test_points + 1))

                # AJUSTE: Usando predict_interval para obter os intervalos de confiança
                pred_interval = model.predict_interval(fh=fh, coverage=0.95)
                forecast_values = pred_interval.iloc[:, 0]  # A primeira coluna é a previsão

                mape = mean_absolute_percentage_error(test_series.values, forecast_values.values) * 100

                results.update({
                    "test_x": [DataUtils.safe_json_serialize(x) for x in test_series.index],
                    "test_y": test_series.values.tolist(),
                    "forecast_x": [DataUtils.safe_json_serialize(x) for x in test_series.index],
                    "forecast_y": forecast_values.values.tolist(),
                    "forecast_ci_lower": pred_interval.iloc[:, 1].values.tolist(),  # Segunda coluna é o limite inferior
                    "forecast_ci_upper": pred_interval.iloc[:, 2].values.tolist(),
                    # Terceira coluna é o limite superior
                    "mape": float(mape)
                })
            else:
                fh = [1, 2, 3, 4]  # Prever 4 períodos no futuro

                # AJUSTE: Usando predict_interval também para o caso sem teste
                pred_interval = model.predict_interval(fh=fh, coverage=0.95)
                forecast_values = pred_interval.iloc[:, 0]

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

                results.update({
                    "forecast_x": [DataUtils.safe_json_serialize(x) for x in forecast_dates],
                    "forecast_y": forecast_values.values.tolist(),
                    "forecast_ci_lower": pred_interval.iloc[:, 1].values.tolist(),
                    "forecast_ci_upper": pred_interval.iloc[:, 2].values.tolist(),
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
            acf_confint_upper = acf_confint[:, 1] - acf_values
            pacf_confint_upper = pacf_confint[:, 1] - pacf_values
            return {
                "lags": list(range(len(acf_values))),
                "acf": acf_values.tolist(), "pacf": pacf_values.tolist(),
                "acf_confidence_upper": acf_confint_upper.tolist(),
                "pacf_confidence_upper": pacf_confint_upper.tolist(),
            }
        except Exception as e:
            logger.warning(f"Erro no cálculo de autocorrelação: {e}")
            return None


# --- Handlers de Erro ---
@app.errorhandler(413)
def file_too_large(error): return jsonify({"error": "Arquivo muito grande. Máximo: 16MB"}), 413


@app.errorhandler(DataProcessingError)
def handle_data_processing_error(error): return jsonify({"error": str(error)}), 400


@app.errorhandler(InsufficientDataError)
def handle_insufficient_data_error(error): return jsonify({"error": str(error)}), 400


# --- Endpoints ---
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
        logger.info(f"Arquivo {file.filename} salvo como {file_id}")
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
        unidades = df['unidade_academica'].unique().tolist() if 'unidade_academica' in df.columns else []
        semestres = df.index.to_series().apply(lambda x: f"{x.year}.{1 if x.month == 1 else 2}").unique().tolist()
        preview_df = df.reset_index().head(100)
        preview_data = {"headers": preview_df.columns.tolist(), "rows": preview_df.astype(str).values.tolist()}
        file_details = FileDetails(unidades=sorted(unidades), semestres=sorted(semestres), preview_data=preview_data)
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
        analyzer = EvasionAnalyzer()
        df = analyzer.load_and_clean_data(json_data=edited_data)
        if 'unidade_academica' in df.columns and params.selected_unidades:
            df_filtered = df[df['unidade_academica'].isin(params.selected_unidades)]
        else:
            df_filtered = df
        time_series = df_filtered.groupby(df_filtered.index)['taxa_evasao'].mean()
        analysis_results = analyzer.perform_analysis(
            data_series=time_series,
            semestre_corte_str=params.selected_semestre
        )
        logger.info("Análise concluída com sucesso")
        return jsonify(analysis_results), 200
    except (DataProcessingError, InsufficientDataError) as e:
        raise e  # Deixa os handlers de erro customizados capturarem
    except Exception as e:
        logger.error(f"Erro na análise: {e}")
        traceback.print_exc()
        return jsonify({"error": "Erro interno do servidor"}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    # AJUSTE: Versão atualizada para refletir as mudanças
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "2.2"})


# --- Execução ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)