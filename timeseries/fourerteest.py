import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def detrend(prices,method='difference'):
 if method=='difference':
     detrended=prices.Close[1:]-prices.Close[:-1].values
     print(detrended)
     
 elif method=='linear':
     X=np.arange(0,len(prices))
     y=prices.Close.values
     model =LinearRegression()
     model.fit(X.reshape(-1,1),y.reshape(-1,1))
     trend=model.predict(X.reshape(-1,1))
     trend = trend.reshape((len(prices),))
     detrended=prices.Close-trend
 else:
     print('You did not have valid method for detrending')
     
 return detrended

df=pd.read_csv('Prices/EURUSD_.csv')
df.columns=['Date','Open','High','Low','Close','Volume']
df['Date']=pd.to_datetime(df['Date'])
df=df.set_index(df.Date)
foresight=20
data=pd.DataFrame()
dataR=pd.DataFrame()
data['Close']=df['Close']
#data=data[:119]
dataR['Close']=df['Close']
training=226



data.reset_index(inplace=True)
dataR.reset_index(inplace=True)
data=data[:training]

#print(data.tail())
#print(dataR.tail())
poly=np.polyfit(data.index,data.Close.values,deg=10)
per_err=np.sum((np.polyval(np.poly1d(np.polyfit(data.index,data.Close.values,deg=10)),data.index)-data.Close.values)**2)
#per_err=np.sum(np.polyval(poly,data.index)-data.Close.values)**2
print('Percentage error:',per_err,'%')

plt.plot(np.poly1d(np.polyval(poly,data.index)),label='Fit')
plt.plot(data.index,data.Close.values,color='r',label='Close')
plt.show()

#diffrence between curve and data
pp1=np.polyval(poly,len(data)+foresight)
diff =data.Close.values-pp1
'''
plt.plot(data.index,diff,color='b',label='difference')
plt.show()'''
#applying fourier transformation

Y=np.fft.fft(diff)
'''
for i in range(0,len(data)):
     if Y[i]>Y[5]*10**-2:
          Y[i]=0
Yhat=np.abs(Y)'''

'''
plt.plot(Yhat)
plt.show()'''

'''

for i in range(0,len(data)):
     if Y[i]>Y[5]:
          Y[i]=0
     '''     

PP=np.fft.ifft(Y)
'''
plt.plot(PP ,color='b')
plt.plot(PP-diff ,color='b')
plt.show()'''
YY={}
a={}
b={}



for n in range(len(data),len(data)+foresight):
      YY[n]=0
      for k in range(0,len(data)):
         a[k]=np.real(Y[k])       
         b[k]=-np.imag(Y[k])
         omk=2*np.pi*(k-1)/len(data)
         YY[n]=YY[n]+a[k]*np.cos(omk*(n-1)+b[k]*np.sin(omk*(n-1)))
         
      
      YY[n]=-YY[n]/len(data)
      
      
print(len(YY))

print(YY)
print('-----------------------------')
print(diff)
print('-----------------------------')
x=0;
XY={}
for i in YY:
     XY[x]=i
     x+=1


YY=np.append(diff,[YY[XY[0]],YY[XY[1]],YY[XY[2]],YY[XY[3]],YY[XY[4]],YY[XY[5]],YY[XY[6]],YY[XY[7]],YY[XY[8]],YY[XY[9]],YY[XY[10]],YY[XY[11]],YY[XY[12]],YY[XY[13]],YY[XY[14]],YY[XY[15]],YY[XY[16]],YY[XY[17]],YY[XY[18]],YY[XY[19]]])
prediction_weight=0
for k in range(training+1,training+foresight):
      prediction_weight+=YY[k]
      if k==foresight+training:
           prediction_weight=prediction_weight/foresight

print(YY)
'''
for i in range(0,len(diff)):
     YY[i]=diff[i]'''


print('The prediction weight is:',prediction_weight)
print('-------------------------------------------------------')

#YY=diff

tot=(pp1)+YY

print(tot)



plt.plot(tot,color='r')
plt.plot(data.index,data.Close.values,color='b')
plt.plot(dataR.index,dataR.Close.values,color='g')

plt.show()


'''
data['Detrend']=detrend(data)
data =data.fillna(0)
data.reset_index(inplace=True)
dt=1
n=len(data)
fhat=np.fft.fft(data.Detrend.values,n)
PSD=fhat*np.conj(fhat)/n
freq=(1/(dt*n))*np.arange(n)
L=np.arange(1,np.floor(n/2),dtype='int')

indices=PSD>100
PSDclean=PSD*indices
fhat=indices*fhat
ffilt=np.fft.ifft(fhat)
fig,axs=plt.subplots(3,1)


plt.sca(axs[0])
plt.plot(data.index,data.Detrend.values,color='r',label='Detrend')
plt.plot()
plt.legend()


plt.sca(axs[1])
plt.plot(freq[L],PSD[L],color='c',label='Noisy')
plt.legend()

plt.sca(axs[2])
plt.plot(freq[L],PSD[L],color='c',label='Noisy')
plt.plot(freq[L],PSDclean[L],color='k',label='Filtered')
plt.legend()


plt.show()



print(data)

'''

