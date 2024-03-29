from fastapi import FastAPI
from Models.PredictionRequest import *
from Models.PredictionResponse import *
from Logics.XGBPrediction import *
from Logics.SVRPrediction import *
from pmdarima.arima import auto_arima
import numpy as np
import pandas as pd
import asyncio
import requests

class PredictionPresenter:
    tsa_df_results: pd.DataFrame
    xgb_df_results: pd.DataFrame
    svr_df_results: pd.DataFrame
    isPredictionFail = False
    isPredictingByXGBComplete = False
    isPredictingBySVRComplete = False
    isPredictingByTSAComplete = False

    async def predict(self, request: PredictionRequest):
        asyncio.create_task(self.predictUsingTSA(request))
        asyncio.create_task(self.predictUsingXGB(request))
        asyncio.create_task(self.predictUsingSVR(request))
        return

    async def predictUsingSVR(self, request: PredictionRequest):
        try:
            self.svr_df_results = svr_approach(request.time_series, request.horizon, request.frequency)
            self.isPredictingBySVRComplete = True
        except:
            self.isPredictionFail = True
        await self.onPredictionComplete(request.predictionId)
        return

    async def predictUsingXGB(self, request: PredictionRequest):
        try:
            self.xgb_df_results = xgboost_approach(request.time_series, request.horizon, request.frequency)
            self.isPredictingByXGBComplete = True
        except:
            self.isPredictionFail = True
        await self.onPredictionComplete(request.predictionId)
        return

    async def predictUsingTSA(self, request: PredictionRequest):
        try:
            # Xử lý dữ liệu và tạo data frame từ Prediction Request
            df_comp = self.convertListToSaleDataFrame(request.time_series, request.frequency)
            # Xử dụng auto sarima để chọn ra model tốt nhất theo AICC
            sarima_model = auto_arima(df_comp,start_q=0, m=1, max_p=20, max_q=0, max_d=2,maxiter=100,n_jobs=-1,alpha=0.05,trace=True, error_action='ignore', information_criterion='aicc', suppress_warnings=True, stepwise= True )
            # Tiến hành dự báo sử dụng model được chọn
            self.tsa_df_results = pd.DataFrame(sarima_model.predict(n_periods = request.horizon))
            self.isPredictingByTSAComplete = True
        except:
            self.isPredictionFail = True
        await self.onPredictionComplete(request.predictionId)
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

    def assemblyTSAandMLPredictionResult(self, tsa_result_df: pd.DataFrame, ml_result_df: pd.DataFrame, svr_result_df: pd.DataFrame)-> pd.DataFrame:
        assembly_df = tsa_result_df
        sales = []
        for i in range(len(assembly_df)):
            sale = (tsa_result_df[0].get(i) + svr_result_df.sale.get(i) * 2 + ml_result_df.sale.get(i) * 3)/6
            sales.append(sale)
        assembly_df[0] = sales
        return assembly_df
    
    def convertDataFrameToListSaleDTO(self, df: pd.DataFrame) -> list:
        results = []
        for i in range(len(df)):
            sale = round(df[0].get(i))
            date = df.index[i].strftime('%Y-%m-%d')
            saleDTO = {'date': date, 'sale': sale}
            results.append(saleDTO)
        return results

    async def onPredictionComplete(self, predictionRecordId: int):
        url = "http://host.docker.internal:8080/api/v1/prediction/process-response"
        results = []
        if(self.isPredictionFail == False):
            if(self.isPredictingByXGBComplete and self.isPredictingByTSAComplete and self.isPredictingBySVRComplete):
                try:
                    df_results = self.assemblyTSAandMLPredictionResult(self.tsa_df_results, self.xgb_df_results, self.svr_df_results)
                    results = self.convertDataFrameToListSaleDTO(df_results)
                except:
                    results = []
            else:
                return
        else:
            results = []
        json = {"predictionId": predictionRecordId, "results": results}
        response = requests.request(method= "PUT",url = url, json = json)
        print(response)
        return

    def onPredictionFail():
        return