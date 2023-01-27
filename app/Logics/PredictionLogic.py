from fastapi import FastAPI
from Models.PredictionRequest import *
from Models.PredictionResponse import *
from Models.SaleDTO import SaleDTO
from pmdarima.arima import auto_arima
import numpy as np
import pandas as pd
import asyncio
import requests

class PredictionPresenter:
    
    async def predict(self, request: PredictionRequest):
        asyncio.create_task(self.predictUsingTSA(request))
        return

    async def predictUsingTSA(self, request: PredictionRequest):
        predictionResponse = PredictionResponse(predictionId=request.predictionId)
        # Xử lý dữ liệu và tạo data frame từ Prediction Request
        df_comp = self.convertListToSaleDataFrame(request.time_series, request.frequency)
        # Xử dụng auto sarima để chọn ra model tốt nhất theo AICC
        sarima_model = auto_arima(df_comp, seasonal=True, m=7, max_p= 5, max_q=5, max_d=2, max_P=5, max_Q=5, max_D=2, n_jobs=-1, error_action='ignore', information_criterion='aicc', suppress_warnings=True, stepwise= False )
        # Tiến hành dự báo sử dụng model được chọn
        df_results = pd.DataFrame(sarima_model.predict(n_periods = request.horizon))
        # Xử lý kết quả dự báo
        predictionResponse.results = self.convertDataFrameToListSaleDTO(df_results)
        await self.onPredictionComplete(predictionResponse)
        return
    
    def convertListToSaleDataFrame(self, list: list, frequency: str) -> pd.DataFrame:
        dates = []
        sales = []
        for i in list:
            dates.append(i.get("date"))
            sales.append(i.get("sale"))
        df_comp = pd.DataFrame({'date': dates, 'sale': sales})
        df_comp.date = pd.to_datetime(df_comp.date, infer_datetime_format=True)
        df_comp.set_index("date", inplace=True)
        df_comp = df_comp.asfreq(frequency)
        return df_comp

    def convertDataFrameToListSaleDTO(self, df: pd.DataFrame) -> list:
        results = []
        for i in range(len(df)):
            sale = int(df[0].get(i))
            date = df.index[i].strftime('%Y-%m-%d')
            saleDTO = {'date': date, 'sale': sale}
            results.append(saleDTO)
        return results

    async def onPredictionComplete(self, predictionResponse: PredictionResponse):
        url = "http://host.docker.internal:8080/api/v1/prediction/process-response"
        json = {"predictionId": predictionResponse.predictionId, "results": predictionResponse.results}
        response = requests.request(method= "PUT",url = url, json = json)
        print(response)
        return

    def onPredictionFail():
        return