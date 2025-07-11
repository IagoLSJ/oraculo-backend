# backend/app.py
import pandas as pd
import numpy as np
import io
import base64
import json
import warnings
from typing import Dict, List, Optional, Tuple, Any

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import matplotlib

matplotlib.use('Agg')  # Importante: Use o backend Agg para que matplotlib funcione em ambiente headless
import matplotlib.pyplot as plt
import seaborn as sns  # Se for usar seaborn para styling ou plots
import os  # Para manipular caminhos de arquivo

# --- Adicione esta constante para o diretório de imagens ---
IMAGE_DIR = 'public_images'  # Nome da pasta onde as imagens serão salvas
# Crie o diretório se ele não existir
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
# ---------------------------------------------------------


app = Flask(__name__)
CORS(app)  # Habilita CORS para todas as rotas

# ====================================================================
# INÍCIO DO CÓDIGO DA CLASSE EVASIONANALYZER (COMPLETO E AJUSTADO)
# ====================================================================

# --- Bibliotecas para análise de séries temporais ---
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.theta import ThetaForecaster
from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

# --- Bibliotecas para análise estatística ---
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA as StatsARIMA


# Classe EvasionAnalyzer (ADAPTADA)
class EvasionAnalyzer:
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None  # Stores the full cleaned DataFrame
        self.processed_data: Dict[str, pd.DataFrame] = {}  # Stores filtered/aggregated DataFrames

    def parse_semester_to_timestamp(self, value: str) -> pd.Timestamp:
        """
        Converts semester format (YYYY.S) to timestamp.
        """
        try:
            value = str(value)
            year, semester = value.split('.')
            year = int(year)
            month = 1 if int(semester) == 1 else 7  # Jan for S1, Jul for S2
            return pd.Timestamp(year=year, month=month, day=1)
        except Exception as e:
            raise ValueError(f"Erro ao converter semestre '{value}': {e}")

    def load_and_clean_data(self, csv_content: str) -> pd.DataFrame:
        """
        Loads and cleans data from CSV content string.
        Performs initial cleaning and type conversion, sets index, but DOES NOT apply asfreq yet.
        """
        try:
            df = pd.read_csv(io.StringIO(csv_content))

            units_to_remove = ['Temporário', 'Itapagé']
            df = df[~df['Unidade'].isin(units_to_remove)]

            columns_to_drop = ['Tx. Retenção (Prazo Padrão)',
                               'Tx. Retenção II (Prazo Máximo)',
                               'Matriculas']
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

            df.rename(columns={
                'Taxa de Evasao': 'taxa_evasao',
                'Taxa de Evasão': 'taxa_evasao',
                'Semestre': 'semestre'
            }, inplace=True)

            df['semestre'] = df['semestre'].apply(self.parse_semester_to_timestamp)

            if df['taxa_evasao'].dtype == 'object':
                df['taxa_evasao'] = pd.to_numeric(
                    df['taxa_evasao'].astype(str).str.replace('%', ''),
                    errors='coerce'
                )
            else:
                df['taxa_evasao'] = pd.to_numeric(df['taxa_evasao'], errors='coerce')

            # Drop rows where critical columns (semestre or taxa_evasao) are NaN after conversion
            df = df.dropna(subset=['semestre', 'taxa_evasao'])

            df.set_index('semestre', inplace=True)
            df.sort_index(inplace=True)

            self.data = df
            print("Dados carregados e limpos:")
            print(df.head())
            return df

        except Exception as e:
            raise Exception(f"Erro ao carregar ou limpar dados: {e}")

    def filter_by_unit(self, df_full: pd.DataFrame, unit_name: str) -> pd.DataFrame:
        """
        Filters data for a specific unit - REMOVIDO asfreq problemático.
        """
        temp_df = df_full.reset_index()  # Reset index to filter by 'Unidade'
        filtered_df = temp_df[temp_df['Unidade'] == unit_name].copy()

        if filtered_df.empty:
            return pd.DataFrame()

        filtered_df = filtered_df.drop(columns=['Unidade'])
        filtered_df.set_index('semestre', inplace=True)

        # Handle potential duplicates in index after filtering
        if not filtered_df.index.is_unique:
            filtered_df = filtered_df[~filtered_df.index.duplicated(keep='first')]

        filtered_df.sort_index(inplace=True)

        # REMOVIDO: asfreq e interpolate que causavam problemas
        # filtered_df = filtered_df.asfreq('6M', fill_value=np.nan).interpolate(method='linear')

        self.processed_data[unit_name] = filtered_df
        print(f"Dados filtrados para {unit_name}:")
        print(filtered_df.head())
        return filtered_df

    def aggregate_by_semester(self, df_full: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates data by semester (sum total) - REMOVIDO asfreq problemático.
        """
        temp_df = df_full.drop(columns=['Unidade'] if 'Unidade' in df_full.columns else [], errors='ignore')
        aggregated = temp_df.groupby(temp_df.index).sum(numeric_only=True)

        # REMOVIDO: asfreq e interpolate que causavam problemas
        # aggregated = aggregated.asfreq('6ME', fill_value=np.nan).interpolate(method='linear')

        self.processed_data['agregado'] = aggregated
        print("Dados agregados:")
        print(aggregated.head())
        return aggregated

    def generate_and_save_decomposition_plot(self, data_series: pd.Series, filename: str):
        """
        Generates and saves the time series decomposition plot.
        Receives a pd.Series (taxa_evasao).
        """
        print(f"Série para decomposição {filename}:")
        print(data_series.head())
        print(f"Valores não-nulos: {data_series.count()}")

        if data_series.empty:
            print(f"Série vazia para decomposição: {filename}")
            return  # Don't raise error, just skip plot generation

        # Drop NaNs before decomposition as it requires non-missing values
        series_clean = data_series.dropna()
        if series_clean.empty:
            print(f"Série vazia após remover NaNs para decomposição: {filename}")
            return
        if len(series_clean) < 4:  # At least 4 points for decomposition
            print(f"Série muito curta para decomposição: {filename}. Mínimo 4 pontos, tem {len(series_clean)}")
            return

        # Ajuste de período para seasonal_decompose (2 para semestral)
        try:
            decomposition = seasonal_decompose(
                series_clean,
                model='additive',
                period=2
            )
        except Exception as e:
            print(f"Erro na decomposição sazonal para {filename}: {e}. Skipping plot.")
            return

        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

        # Plotar com datas no eixo X
        decomposition.observed.plot(ax=axes[0], title='Série Original', color='blue')
        decomposition.trend.plot(ax=axes[1], title='Tendência', color='green')
        decomposition.seasonal.plot(ax=axes[2], title='Sazonalidade', color='orange')
        decomposition.resid.plot(ax=axes[3], title='Resíduos', color='red')

        for ax in axes:
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.tick_params(axis='x', rotation=45)
            ax.set_xlabel('')
            ax.set_ylabel('Taxa (%)')

        plt.suptitle("Decomposição da Série Temporal", y=1.02, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(os.path.join(IMAGE_DIR, filename), bbox_inches='tight', dpi=150)
        plt.close(fig)

    def generate_and_save_acf_pacf_plot(self, data_series: pd.Series, filename: str):
        """
        Generates and saves ACF and PACF plots.
        Receives a pd.Series (taxa_evasao).
        """
        print(f"Série para ACF/PACF {filename}:")
        print(data_series.head())
        print(f"Valores não-nulos: {data_series.count()}")

        if data_series.empty:
            print(f"Série vazia para ACF/PACF: {filename}")
            return

        series_clean = data_series.dropna()
        if series_clean.empty:
            print(f"Série vazia após remover NaNs para ACF/PACF: {filename}")
            return

        # Calcula o número máximo de lags, evitando erro se a série for muito curta
        lags = min(len(series_clean) // 2 - 1, 10)  # Máximo de 10 lags como no protótipo

        if lags < 1:
            print(f"Série muito curta para ACF/PACF: {filename}. Lags={lags}. Mínimo 2 pontos.")
            return

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # ACF
        plot_acf(series_clean, lags=lags, ax=axes[0], title='Autocorrelação (ACF)')
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].set_ylim([-1, 1])

        # PACF
        plot_pacf(series_clean, lags=lags, ax=axes[1], title='Autocorrelação Parcial (PACF)')
        axes[1].grid(True, linestyle='--', alpha=0.6)
        axes[1].set_ylim([-1, 1])

        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR, filename), bbox_inches='tight', dpi=150)
        plt.close(fig)

    def generate_and_save_prediction_plot(self, data_series: pd.Series, filename: str,
                                          order: Tuple[int, int, int] = (1, 1, 1),  # Ordem mais simples
                                          forecast_steps: int = 4):  # Menos passos de previsão
        """
        Generates and saves the Prediction plot using ARIMA model.
        Receives a pd.Series (taxa_evasao).
        """
        print(f"Série para previsão {filename}:")
        print(data_series.head())
        print(f"Valores não-nulos: {data_series.count()}")

        if data_series.empty:
            print(f"Série vazia para previsão: {filename}")
            return

        series_clean = data_series.dropna()
        min_points = max(sum(order) + 2, 5)  # Mínimo mais conservador

        if len(series_clean) < min_points:
            print(
                f"Série muito curta para previsão ARIMA: {filename}. Tamanho={len(series_clean)}. Mínimo {min_points} pontos.")
            return

        try:
            model = StatsARIMA(series_clean, order=order)
            model_fit = model.fit()

            # Predict into the future `forecast_steps`
            forecast_results = model_fit.get_forecast(steps=forecast_steps)
            y_pred = forecast_results.predicted_mean
            conf_int = forecast_results.conf_int(alpha=0.05)

            plt.figure(figsize=(12, 7))

            # Plot original series
            plt.plot(series_clean.index, series_clean, label="Série Original", color='blue', linewidth=2)

            # Plot prediction
            plt.plot(y_pred.index, y_pred, label="Previsão", color='red', linestyle='--', linewidth=2)

            # Confidence Interval
            plt.fill_between(
                y_pred.index,
                conf_int.iloc[:, 0],
                conf_int.iloc[:, 1],
                color='gray',
                alpha=0.2,
                label="IC 95%"
            )

            plt.title("Previsão de Taxa de Evasão")
            plt.xlabel("Semestre")
            plt.ylabel("Taxa de Evasão (%)")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGE_DIR, filename), bbox_inches='tight', dpi=150)
            plt.close()

        except Exception as e:
            print(f"Erro na previsão ARIMA para {filename}: {e}. Skipping plot.")
            # Do not re-raise to avoid breaking the API endpoint for other plots


# ====================================================================
# FIM DO CÓDIGO DA CLASSE EVASIONANALYZER
# ====================================================================


@app.route('/analyze', methods=['POST'])
def analyze_data():
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({'error': 'Requisição JSON inválida.'}), 400

        csv_content_b64 = request_data.get('fileContent')
        # Filters are now expected as a list of unit names (e.g., ['QUIXADÁ'])
        filters = request_data.get('filters', [])
        max_semester = request_data.get('maxSemester')  # Semestre máximo para treinar (e.g., '2019.1')

        if not csv_content_b64:
            return jsonify({'error': 'Conteúdo do arquivo CSV não fornecido.'}), 400

        csv_content = base64.b64decode(csv_content_b64).decode('utf-8')

        analyzer = EvasionAnalyzer()

        df_full = analyzer.load_and_clean_data(csv_content)

        if df_full.empty:
            return jsonify({'error': 'Não foi possível carregar ou processar os dados do CSV.'}), 400

        # Determine the series to use for plotting based on filters
        data_series_for_plots: pd.Series

        if filters:
            # Assuming 'Unidade' column exists in df_full
            # Filter df_full by selected units first, then aggregate for the series
            if 'Unidade' in df_full.columns:
                df_filtered_by_units = df_full[df_full['Unidade'].isin(filters)].copy()
                if df_filtered_by_units.empty:
                    return jsonify(
                        {'error': f"Nenhum dado encontrado para as unidades filtradas: {', '.join(filters)}"}), 400

                # Aggregate across selected units for the analysis series
                data_series_for_plots = df_filtered_by_units.groupby(df_filtered_by_units.index)[
                    'taxa_evasao'].mean()  # Using mean for multiple units

                print("Série filtrada e agregada:")
                print(data_series_for_plots.head())

            else:
                warnings.warn(
                    "Coluna 'Unidade' não encontrada no DataFrame para aplicar filtros. Usando dados agregados.")
                data_series_for_plots = analyzer.aggregate_by_semester(df_full)['taxa_evasao']
        else:
            # If no units are selected for filtering, use the overall aggregated data
            aggregated_df = analyzer.aggregate_by_semester(df_full)
            if aggregated_df.empty or 'taxa_evasao' not in aggregated_df.columns:
                return jsonify(
                    {'error': 'Não foi possível agregar dados por semestre ou coluna taxa_evasao não encontrada.'}), 400
            data_series_for_plots = aggregated_df['taxa_evasao']

        # Apply max_semester filter to the selected series
        if max_semester:
            try:
                max_sem_ts = analyzer.parse_semester_to_timestamp(max_semester)
                data_series_for_plots = data_series_for_plots[data_series_for_plots.index <= max_sem_ts].copy()
                if data_series_for_plots.empty:
                    return jsonify({'error': f"Nenhum dado após o filtro de semestre máximo: {max_semester}"}), 400
            except ValueError as e:
                print(f"Aviso: Semestre máximo inválido: {e}. Ignorando filtro de semestre.")

        # Final check before passing to plot functions
        if data_series_for_plots.empty:
            return jsonify({'error': 'Nenhum dado válido para gerar os gráficos após todos os filtros.'}), 400

        print("Série final para análise:")
        print(data_series_for_plots.head())
        print(f"Total de valores: {len(data_series_for_plots)}")
        print(f"Valores não-nulos: {data_series_for_plots.count()}")

        # --- Salvar os gráficos como imagens ---
        import uuid
        analysis_id = str(uuid.uuid4())  # Unique ID for this analysis

        decomposition_image_name = f'decomposicao_{analysis_id}.png'
        acf_pacf_image_name = f'acf_pacf_{analysis_id}.png'
        prediction_image_name = f'predicao_{analysis_id}.png'

        analyzer.generate_and_save_decomposition_plot(data_series_for_plots, decomposition_image_name)
        analyzer.generate_and_save_acf_pacf_plot(data_series_for_plots, acf_pacf_image_name)
        analyzer.generate_and_save_prediction_plot(data_series_for_plots, prediction_image_name)

        # --- Retornar as URLs das imagens geradas ---
        response_data = {
            "image_urls": {
                "decomposicao": f"/images/{decomposition_image_name}",
                "acf_pacf": f"/images/{acf_pacf_image_name}",
                "predicao": f"/images/{prediction_image_name}",
            },
            "message": "Análise e geração de imagens concluídas com sucesso!"
        }

        return jsonify(response_data), 200

    except Exception as e:
        print(f"Erro na execução da análise: {e}")
        return jsonify({'error': str(e)}), 500


# --- Rota para servir as imagens estáticas ---
@app.route('/images/<path:filename>')
def serve_image(filename):
    """
    Serves static files from the IMAGE_DIR directory.
    """
    return send_from_directory(IMAGE_DIR, filename)


if __name__ == '__main__':
    # Ensure the image directory exists when running locally
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    app.run(debug=True, host='0.0.0.0', port=5000)