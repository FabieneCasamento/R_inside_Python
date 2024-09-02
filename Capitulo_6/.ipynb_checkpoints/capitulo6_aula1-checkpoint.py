# Databricks notebook source
# MAGIC %md
# MAGIC Modelos estatísticos lineares para séries temporais. Estão relacionados à regressão linear, porém representam as correlações entre os pontos de dados na mesma série temporal. Exitem os:
# MAGIC
# MAGIC - Modelos autorregressivos (AR), modelos de média móvel (MA) e modelos autorregressivos de média móvel (ARIMA). 
# MAGIC
# MAGIC - Autorregressão vertical (VAR).
# MAGIC
# MAGIC - Modelos hierárquicos. 
# MAGIC
# MAGIC Que são utilizados para a previsão de sérires temporais.

# COMMAND ----------

# MAGIC %md
# MAGIC Regressão Linear presume que você tem dados independentes e indenticamente distribuídos (iid).
# MAGIC
# MAGIC Mas, em dados de séries temporais isso não ocorre. O que acontece na ST é que os pontos próximos no tempo costumam estar fortemente correlacionados uns com os outros. Na realidade, quando não há correlações temporais, os dados de séries temporais dificilmente servem para tarefas tradicionais de séries temporais, como predizer o futuro ou compreender a dinâmica temporal. 
# MAGIC
# MAGIC Não raro, os tutorais e os livros de séries temporais nos passam a impressão indevida de que a regressão linear não serve para séries temporais. O que faz os alunos acreditarem que regressões lineares simples não são suficientes. Mas não é assim que funciona. <b> A regressão linear de mínimos quadrados ordinários pode ser aplicada aos dados da série temporal desde que as seguintes condições sejam atendidas:</b>
# MAGIC
# MAGIC
# MAGIC Suposições ao comportamento das séries temporais:
# MAGIC
# MAGIC - A séire temporal tem uma resposta linear aos seus preditores. 
# MAGIC - nenhuma variável de entrada é constante ao longo do tempo ou perfeitamente correlacionada com outra variável de entrada. Assim, o tradicional requisito de regressão linear de variáveis independentes se amplifica para considerar a dimensão temporal dos dados. 
# MAGIC
# MAGIC Suposições a respeito do erro:
# MAGIC
# MAGIC - Para cada ponto no tempo, o valor esperado do erro, dadas todas as variáveis explicativas para todos os períodos de tempo (passo à frente [forward] e passo para trás [backward]), é 0.
# MAGIC
# MAGIC - o erro em qualquer período de tempo determinado não se correlaciona com as entradas em nenhum período de tempo no passado ou futuro. Portanto, um gráfico da função de autocorrelação dos erros não indicará nenhum padrão. 
# MAGIC
# MAGIC - a variância do erro independe do tempo. 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC se for válido, a regressão de mínimos quadrados ordinários é um estimado não viesado dos coeficientes dos dados de entrada, mesmo em séries temporais. Nesse caos, as variâncias das estimativas amostradas têm a mesma forma matemática para a regressão linear padrão. Logo, se seus dados atenderem às suposições enumeradas, dá para aplicar a regressão linear. Dados sempelhates aos da regressão linear padrão aplicada a dados transversais. 
# MAGIC
# MAGIC Não force a regressão linear:
# MAGIC
# MAGIC - Seus coeficientes não minimizarão o ero de seu modelo. 
# MAGIC
# MAGIC - Seus valores-p para determinar se seus coeficientes são diferentes de zero estarão incorretos, pois eles se baseiam em suposições ue não são atendidas. Isso significaque suas avaliações da significância do coeficiente podem estar errados. 
# MAGIC
# MAGIC
# MAGIC
# MAGIC se uma estimativa não for superestimada ou subestimara, o estimador é não viesado. Isso costuma ser bom, embora tende estar atento ao trade-off de viés/variância, a representação dos problemas estatísticos e de aprendizado de máquina em que modelos com um viés inferior em sua estimativas de parâmetros tendem a ter uma maior variância dessa mesma estimativa. A variância da estimativa do parâmetro retrata o nível de variância de uma estimativa em diferentes amostras de dados. 

# COMMAND ----------

# MAGIC %md
# MAGIC https://github.com/PracticalTimeSeriesAnalysis/BookRepo/tree/master/Ch07

# COMMAND ----------

# MAGIC %md
# MAGIC Regressão ok, o pacorte tsml para função de forecast é válida 
# MAGIC
# MAGIC https://robjhyndman.com/hyndsight/forecast7-part-2/

# COMMAND ----------

!pip install rpy2

# COMMAND ----------

import rpy2
print(rpy2.__version__)

# COMMAND ----------

import rpy2.robjects as robjects


from rpy2.robjects.packages import importr
# import R's "base" package
base = importr('base')

# import R's "utils" package
utils = importr('utils')

# COMMAND ----------


robjects.r("install.packages('forecast', repos = 'https://cran.r-project.org/')")



# COMMAND ----------

codigo = """
#install.packages('forecast')
library(forecast)
library(ggplot2)

png(filename = "Figura1.png",width = 10, height = 8, 
     units = "cm",pointsize = 12, res = 1200)

deaths.lm  <- tslm(mdeaths ~ trend + fourier(mdeaths,3))
mdeaths.fcast <- forecast(deaths.lm,
    data.frame(fourier(mdeaths,3,36)))
autoplot(mdeaths.fcast)

dev.off()
"""

robjects.r(codigo)

# COMMAND ----------

codigo = """
library(forecast)
library(ggplot2)


png(filename = "Figura1.png",width = 10, height = 8, 
     units = "cm",pointsize = 12, res = 1200)


deaths.lm  <- tslm(mdeaths ~ trend + fourier(mdeaths,3))
mdeaths.fcast <- forecast(deaths.lm,
    data.frame(fourier(mdeaths,3,36)))
autoplot(mdeaths.fcast)

dev.off()
"""

robjects.r(codigo)

# COMMAND ----------

codigo = """
library(forecast)
library(ggplot2)

png(filename = "Figura1.png",width = 10, height = 8, 
     units = "cm",pointsize = 12, res = 1200)

deaths.lm <- tslm(mdeaths ~ trend + fourier(mdeaths,3))
mdeaths.fcast <- forecast(deaths.lm,
                          data.frame(fourier(mdeaths,3,36)))

plot(mdeaths.fcast)

dev.off()
"""

robjects.r(codigo)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC <img src="Figura1.png" />
# MAGIC
# MAGIC
# MAGIC <img src="Figura1.png" width="720" height="500" />
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Note that fourier now takes 3 arguments. The first is the series, which is only used to grab the seasonal period and the tsp attribute. The second argument K is the number of Fourier harmonics to compute. If the third argument h is NULL (the default), the function returns Fourier terms for the times of the historical observations. But if h is a positive integer, the function returns Fourier terms for the next h time periods after the end of the historical data.
# MAGIC
# MAGIC The lm function has long allowed a matrix to be passed and independent linear models fitted to each column. The new tslm function also allows this now.

# COMMAND ----------

# MAGIC %md
# MAGIC # Bias adjustment for Box-Cox transformations
# MAGIC
# MAGIC Almost all modelling and forecasting functions in the package allow Box-Cox transformations to be applied before the model is fitted, and for the forecasts to be back transformed. This will give median forecasts on the original scale, as I’ve explained before.
# MAGIC
# MAGIC There is now an option to adjust the forecasts so they are means rather than medians, but setting biasadj=TRUE whenever the forecasts are computed. I will probably make this the default in some future version, but for now the default is biasadj=FALSE so the forecasts are actually medians.

# COMMAND ----------

robjects.r("install.packages('fpp', repos = 'https://cran.r-project.org/')")


# COMMAND ----------



robjects.r("install.packages('ggplot2', repos = 'https://cran.r-project.org/')")


# COMMAND ----------

codigo = """
# Install and load the required package
if (!require(fpp)) {
  install.packages("fpp")
}
library(fpp)
"""
robjects.r(codigo)


# COMMAND ----------

#não rodou
codigo = """

png(filename = "Figura2.png")

library(fpp, quietly=TRUE)

fit <- ets(eggs, model="AAN", lambda=0)
fc1 <- forecast(fit, biasadj=TRUE, h=20, level=95)
fc2 <- forecast(fit, biasadj=FALSE, h=20)
cols <- c("Mean"="#0000ee","Median"="#ee0000")
autoplot(fc1) + ylab("Price") + xlab("Year") +
  autolayer(fc2, PI=FALSE, series="Median") +
  autolayer(fc1, PI=FALSE, series="Mean") +
  guides(fill=FALSE) +
  cscale_colour_manual(name="Forecasts",values=cols)





dev.off()
"""
#robjects.r(codigo)

# COMMAND ----------

codigo = """ png(filename = "Figura2.png")

library(fpp, quietly=TRUE)
library(ggplot2)

fit <- ets(eggs, model="AAN", lambda=0)
fc1 <- forecast(fit, biasadj=TRUE, h=20, level=95)
fc2 <- forecast(fit, biasadj=FALSE, h=20)
cols <- c("Mean"="#0000ee","Median"="#ee0000")

autoplot(fc1) + ylab("Price") + xlab("Year") +
  autolayer(fc2, PI=FALSE, series="Median") +
  autolayer(fc1, PI=FALSE, series="Mean") +
  guides(fill=FALSE) +
  scale_colour_manual(name="Forecasts",values=cols) 

dev.off()"""
robjects.r(codigo)

# COMMAND ----------

codigo = """ png(filename = "Figura2.png")

library(fpp, quietly=TRUE)
library(ggplot2)

fit <- ets(eggs, model="AAN", lambda=0)
fc1 <- forecast(fit, biasadj=TRUE, h=20, level=95)
fc2 <- forecast(fit, biasadj=FALSE, h=20)
cols <- c("Mean"="#0000ee","Median"="#ee0000")

plot(autoplot(fc1) + ylab("Price") + xlab("Year") +
  autolayer(fc2, PI=FALSE, series="Median") +
  autolayer(fc1, PI=FALSE, series="Mean") +
  guides(fill=FALSE) +
  scale_color_manual(name="Forecasts",values=cols)) # Correção: Use scale_color_manual

dev.off()
"""
robjects.r(codigo)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC <img src="Figura2.png" width="720" height="500" />

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="Figura2.png" />

# COMMAND ----------

# MAGIC %md
# MAGIC A new Ccf function
# MAGIC Cross-correlations can now be computed using Ccf, mimicing ccf except that the axes are more informative.
# MAGIC
# MAGIC The Acf function now handles multivariate time series, with cross-correlation functions computed as well as the ACFs of each series.
# MAGIC
# MAGIC Covariates in neural net AR models
# MAGIC The nnetar function allows neural networks to be applied to time series data by building a nonlinear autoregressive model. A new feature allows additional inputs to be included in the model.
# MAGIC
# MAGIC Better subsetting of time series
# MAGIC subset.ts allows quite sophisticated subsetting of a time series. For example

# COMMAND ----------

codigo = """ png(filename = "Figura3.png")

plot(subset(gas,month="November"))


dev.off()#para realmente gerar a figura
"""
robjects.r(codigo)

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="Figura3.png" />

# COMMAND ----------

codigo = """ subset(woolyrnq,quarter=3)
"""
robjects.r(codigo)

# COMMAND ----------

codigo = """ print(subset(woolyrnq,quarter=3))

print("This is now substantially more robust than it used to be.")
"""
robjects.r(codigo)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ### Modelo Autorregressivos (AR)
# MAGIC Modelo Autorregressivos (AR) tomoa como base a inutição de que o passado prediz o futuro. Desse modo, ele pressupõe um processo de série temporal no qual o valor em um ponto no tempo t é uma função dos valores da série em pontos anteriores no tempo. 
# MAGIC
# MAGIC  Autorregressão seroa uma regressão em valores passados para predizer valores futuros. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC AR(1) se assemelha a regressão simples
# MAGIC
# MAGIC y_t = b0 +b1*y_t-1 +e_t
# MAGIC
# MAGIC
# MAGIC y_t = série no tempo t
# MAGIC b0= constante
# MAGIC b1*y_t-1 = seu valor no intervalo de tempo anterior multiplicado por outra constante
# MAGIC e_t= termo de erro que varia no tempo 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Probabilidade condicional
# MAGIC E(y_t| y_t-1) = b0 + b1*y_t-1 + e_t
# MAGIC
# MAGIC
# MAGIC
# MAGIC A <b>estacionariedade</b> é um conceito fundamental na análise de séries temporais, já que é exigência de muitos modelos de séries temporais, incluindo modelos AR. Determinamos um modelo autorregressivo (AR) com condições ser estacionário a partir da definição de estacionariedade.

# COMMAND ----------

# MAGIC %md
# MAGIC resultados var(y_t) = var(e1)/ (1- (Ø_1)^2)
# MAGIC
# MAGIC considerando que a variância deve ser maior ou igual a 0 por definição, podemos ver que  (Ø_1)^2 deve ser menor que 1 para garantir um valor positivo no lado direito da equação anterior. Ou seja, para um processo estacionário, devemos ter -1 < Ø_1 < 1, o que é uma condição necessária e suficiente para esse tipo de estacionariedade fraca. 

# COMMAND ----------


