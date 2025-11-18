import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score
    
 

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are you having a Heart Attack or not?❤️‍🩹")
    st.sidebar.markdown("Are you having a Heart Attack or not?❤️‍🩹")
    
    @st.cache_data(persist=True) 
    def load_data():
        data = pd.read_csv('Heart Attack.csv')
        label = LabelEncoder()
        for col in data.columns:
          data[col] = label.fit_transform(data[col])
        return data
 
 
    @st.cache_data(persist=True)
    def split(df):
      y = df.classifier
      x = df.drop(columns=['classifier'])
      x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
      return x_train, x_test, y_train, y_test
    
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['Positive','negative']
    
    if st.sidebar.checkbox("Show Raw Data",False):
        st.subheader("Heart Attack Data set(Classification)")
        st.write(df)
    def plot_metrics(metrics_list):
       if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, model.predict(x_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        st.pyplot(fig)
        #if 'ROC CURVE' in metrics_list:
            #st.subheader("ROC Curve")
            
            
        #if 'Precision-Recall Curve' in metrics_list:
            #st.subheader("Precision Recall Curve")
            
            
    st.sidebar.subheader("Choose classifier:")
    classifier=st.sidebar.selectbox("Classifier",("Support Vector Machine(SVM)","Logistic Regression"))
    
    if classifier == 'Support Vector Machine(SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        c=st.sidebar.number_input("C (Regularization paramter)",0.01,10.0,step=0.01,key='c')
        kernel=st.sidebar.radio("kernel",("rbf","linear"),key='kernel')
        gamma=st.sidebar.radio("Gamma(Kernel coefficient)",("scale","auto"),key='gamma')
        
        metrics=st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix','ROC CURVE','Precision-Recall CURVE'))
        if st.sidebar.button("classify",key='classify'):
            st.subheader("Support Vector Machine(SVM) Results:")
            model=SVC(C=c,kernel=kernel,gamma=gamma) 
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            test_pred={'age':[42],'gender':[1],'impluse':[75],'pressurehight':[63],'pressurelow':[40],'glucose':[210],'kcm':[540],'troponin':[322]}
            test=pd.DataFrame(test_pred)
            predictions=model.predict(test)
            st.write("Model Accuracy is:",accuracy.round(2))
            st.write("The Predicted Class Is:",predictions.round(2))
            st.write("Precision:", precision_score(y_test, model.predict(x_test), labels=class_names).round(2))
            st.write("RECALL:", recall_score(y_test, model.predict(x_test), labels=class_names).round(2))
            plot_metrics(metrics)
    
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        c=st.sidebar.number_input("C (Regularization paramter)",0.01,10.0,step=0.01,key='c_RL')
        max_iter = st.sidebar.slider("Maximum number of iteration",100,500,key='max_iter')
                
        metrics=st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix','ROC CURVE','Precision-Recall CURVE'))
        if st.sidebar.button("classify",key='classify'):
            st.subheader("Logistic Regression Results:")
            model=LogisticRegression(C=c,max_iter=max_iter) 
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            test_pred={'age':[60],'gender':[0],'impluse':[110],'pressurehight':[90],'pressurelow':[50],'glucose':[150],'kcm':[540],'troponin':[322]}
            test=pd.DataFrame(test_pred)
            predictions=model.predict(test)
            st.write("Model Accuracy is:",accuracy.round(2))
            st.write("The Predicted Class Is:",predictions.round(2))
            st.write("Precision:", precision_score(y_test, model.predict(x_test), labels=class_names).round(2))
            st.write("RECALL:", recall_score(y_test, model.predict(x_test), labels=class_names).round(2))
            plot_metrics(metrics)

            
        




if __name__ == "__main__":
    main()