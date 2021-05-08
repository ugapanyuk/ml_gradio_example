import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

def load_data():
    '''
    Загрузка данных
    '''
    data = pd.read_csv('data/occupancy_datatraining.txt', sep=",")
    return data

def preprocess_data(data_in):
    '''
    Масштабирование признаков, функция возвращает X и y для кросс-валидации
    '''
    data_out = data_in.copy()
    # Числовые колонки для масштабирования
    scale_cols = ['Temperature', 'Humidity', 'Light', 'CO2']
    new_cols = []
    sc1 = MinMaxScaler()
    sc1_data = sc1.fit_transform(data_out[scale_cols])
    for i in range(len(scale_cols)):
        col = scale_cols[i]
        new_col_name = col + '_scaled'
        new_cols.append(new_col_name)
        data_out[new_col_name] = sc1_data[:,i]
    return data_out[new_cols], data_out['Occupancy']

data = load_data()
data_X, data_y = preprocess_data(data)

def knn(cv_knn, cv_slider):
    '''
    Входы и выходы функции соединены с компонентами в интерфейсе
    '''
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=cv_knn), 
        data_X, data_y, scoring='accuracy', cv=cv_slider)
    scores_dict = {str(i):float(scores[i]) for i in range(len(scores)) }
    return scores_dict, scores_dict, np.mean(scores)

#Входные компоненты
cv_knn = gr.inputs.Slider(minimum=1, maximum=1000, step=1, default=5, label='Количество соседей')
cv_slider = gr.inputs.Slider(minimum=3, maximum=20, step=1, default=5, label='Количество фолдов') 

#Выходные компоненты
out_folds_label = gr.outputs.Label(type='confidences', label='Оценки по фолдам (Label)')
out_folds_kv = gr.outputs.KeyValues(label='Оценки по фолдам (KeyValues)')
out_acc = gr.outputs.Textbox(type="number", label='Усредненное значение accuracy')

iface = gr.Interface(
  fn=knn, 
  inputs=[cv_knn, cv_slider], 
  outputs=[out_folds_label, out_folds_kv, out_acc],
  title='Метод ближайших соседей')
iface.launch()

