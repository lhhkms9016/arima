import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# Set the page configuration to wide
st.set_page_config(layout="wide")

# Streamlit 앱 제목
st.title('Bitcoin Price Prediction using ARIMA')

# 데이터 파일 업로드
file_path = st.file_uploader("Upload the Bitcoin market price CSV file", type="csv")

# 사이드바에서 예측 기간 입력
forecast_steps = st.sidebar.number_input('Forecast Period (days)', min_value=1, max_value=30, value=5)

if file_path:
    bitcoin_df = pd.read_csv(file_path, names=['day', 'price'])

    # to_datetime으로 day 피처를 시계열 피처로 변환합니다. 
    bitcoin_df['day'] = pd.to_datetime(bitcoin_df['day'])

    # day 데이터를 데이터프레임의 index로 설정합니다.
    bitcoin_df.set_index('day', inplace=True)
    
    st.subheader('Bitcoin Data Overview')
    st.write(bitcoin_df.head())
    st.write(bitcoin_df.describe())

    # 일자별 비트코인 시세를 시각화합니다.
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Bitcoin Price Over Time')
        fig, ax = plt.subplots()
        bitcoin_df.plot(ax=ax)
        st.pyplot(fig)
    
    # ARIMA 모델 학습
    model = ARIMA(bitcoin_df['price'].values, order=(2, 1, 2))
    model_fit = model.fit()
    with col2:
        st.subheader('ARIMA Model Summary')
        st.text(model_fit.summary())

    # 학습 데이터에 대한 예측 결과를 시각화합니다.
    col3, col4 = st.columns(2)
    with col3:
        st.subheader('Training Data Predictions')
        fig, ax = plt.subplots()
        start_index = 0
        end_index = len(bitcoin_df) - 1
        pred = model_fit.get_prediction(start=start_index, end=end_index)
        pred_mean = pred.predicted_mean
        pred_ci = pred.conf_int()

        ax.plot(bitcoin_df.index[start_index:end_index+1], bitcoin_df['price'].values[start_index:end_index+1], label='Observed')
        ax.plot(bitcoin_df.index[start_index:end_index+1], pred_mean, color='r', label='Forecast')
        ax.fill_between(bitcoin_df.index[start_index:end_index+1], pred_ci[:, 0], pred_ci[:, 1], color='pink', alpha=0.3)
        ax.set_title("Training Data Predictions")
        ax.legend()
        st.pyplot(fig)
    
    with col4:
        st.subheader('Residuals')
        residuals = pd.DataFrame(model_fit.resid)
        fig, ax = plt.subplots()
        residuals.plot(title="Residuals", ax=ax)
        st.pyplot(fig)

    # 사이드바에서 테스트 데이터 입력
    st.sidebar.subheader('Enter Test Data')
    
    last_day = bitcoin_df.index[-1]
    test_days = pd.date_range(start=last_day + pd.Timedelta(days=1), periods=forecast_steps)

    test_data_df = st.sidebar.data_editor(
        pd.DataFrame(
            {
                'day': test_days.strftime('%Y-%m-%d').tolist(),
                'price': [0.0] * forecast_steps  # 초기값 0.0으로 설정, 사용자가 직접 입력하도록 유도
            }
        ),
        num_rows="dynamic",
        use_container_width=True
    )

    # 학습 데이터셋으로부터 예측 기간만큼 예측합니다.
    forecast_data = model_fit.get_forecast(steps=forecast_steps)

    if not test_data_df.empty:
        # 입력된 테스트 데이터를 데이터프레임으로 처리
        test_data_df['day'] = pd.to_datetime(test_data_df['day'])
        test_data_df.set_index('day', inplace=True)
        
        # 예측 데이터와 실제 데이터를 비교합니다.
        pred_y = forecast_data.predicted_mean.tolist()  # 예측 데이터
        pred_ci = forecast_data.conf_int()
        pred_y_lower = pred_ci[:, 0].tolist()  # 예측 데이터의 최소값
        pred_y_upper = pred_ci[:, 1].tolist()  # 예측 데이터의 최대값

        test_y = test_data_df['price'].values[:forecast_steps]  # 실제 예측 기간만큼의 가격 데이터

        # 예측과 실제 데이터를 시각화합니다.
        col5, col6 = st.columns(2)
        with col5:
            st.subheader('Forecast vs Actuals')
            fig, ax = plt.subplots()
            ax.plot(test_days, test_y, label='Actual', marker='o')
            ax.plot(test_days, pred_y, label='Forecast', marker='x')
            ax.fill_between(test_days, pred_y_lower, pred_y_upper, color='pink', alpha=0.3)
            ax.set_title("Forecast vs Actuals")
            ax.legend()
            st.pyplot(fig)

        with col6:
            st.subheader('Prediction vs Actuals Detailed View')
            fig, ax = plt.subplots()
            ax.plot(test_days, pred_y, color="gold", label="Predicted Price")  # 모델이 예상한 가격 그래프입니다.
            ax.plot(test_days, pred_y_lower, color="red", linestyle='dashed', label="Lower Confidence Interval")  # 모델이 예상한 최소가격 그래프입니다.
            ax.plot(test_days, pred_y_upper, color="blue", linestyle='dashed', label="Upper Confidence Interval")  # 모델이 예상한 최대가격 그래프입니다.
            ax.plot(test_days, test_y, color="green", label="Actual Price")  # 실제 가격 그래프입니다.
            ax.legend()
            st.pyplot(fig)

        # 모델 성능 평가
        rmse = sqrt(mean_squared_error(test_y, pred_y))
        r2 = r2_score(test_y, pred_y)
        st.subheader('Model Performance')
        st.write(f'Root Mean Squared Error (RMSE): {rmse}')
        st.write(f'R-squared (R²): {r2}')

        # 예측과 실제 데이터 표시
        result_df = pd.DataFrame({
            'day': test_days,
            'actual': test_y,
            'predict': pred_y
        })
        st.subheader('Prediction vs Actuals Data')
        st.dataframe(result_df)
    else:
        st.write("Please enter the test data in the sidebar.")
else:
    st.write("Please upload the required CSV file for the Bitcoin market price.")

