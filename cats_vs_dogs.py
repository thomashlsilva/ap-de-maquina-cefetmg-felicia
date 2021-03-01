from keras.preprocessing.image import ImageDataGenerator,array_to_img
from keras import Model
from keras import layers
from keras import Input
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from keras.models import load_model
import math
import os.path



class Constantes():
    #endereço dos arquivos de treino, validação e teste
    ARR_STR_DATA_DIR = ["data/treino",
                        "data/validacao",
                        "data/teste"]

    #posição do endereço de treino, validação e teste dentro do array ARR_STR_DATA_DIR
    IDX_TREINO = 0
    IDX_VALIDACAO = 1
    IDX_TESTE =2

    #seed para auxiliar a reprodutibilidade da prática
    SEED = 2

    #a quantidade de treino, validação e teste
    QTD_TREINO = 2048
    QTD_VALIDACAO = 1024
    QTD_TESTE = 1024

class ParametrosRedeNeural():
    def __init__(self,int_batch_size=64,
                 int_num_steps_per_epoch=None,
                 int_num_epochs=32,
                optimizer=None):
        self.int_batch_size = int_batch_size
        self.int_num_epochs = int_num_epochs

        #Temos que colocar o numero de passos em uma epoca o suficiente para
        #percorrer o treino todo.
        if not int_num_steps_per_epoch:
            self.int_num_steps_per_epoch = math.ceil(Constantes.QTD_TREINO/int_batch_size)
        else:
            self.int_num_steps_per_epoch = int_num_steps_per_epoch

        #Define qual otimizador será usando (no objeto do otimizador, é definido também o learning rate)
        if not optimizer:
            self.optimizer = RMSprop(learning_rate=0.001, rho=0.9)
        else:
            self.optimizer = optimizer





def plot_imgs_from_iterator(it_datagen,num_lines,num_cols):
    i = 0
    bolFirst = True
    plt.figure(figsize=(9,2*num_lines))
    for mat_x,arr_y in it_datagen:
        if bolFirst:
            print(f'x treino shape: {mat_x.shape}')
            print(f'y treino shape: {arr_y.shape}')
            bolFirst = False

        for idx_img in range(mat_x.shape[0]):

            plt.subplot(num_lines,num_cols,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(array_to_img(mat_x[idx_img]))
            plt.xlabel(f"Classe: {arr_y[idx_img]}")
            i += 1
        if(i>(num_lines*num_cols)-1):
            break
    plt.show()

def get_dataset(param_training,arr_str_data_dir):
    #cria um vetor de 3 objetos ImageDataGenerator(rescale=1/255) - um para cada partição
    arr_obj_datagen = [ImageDataGenerator(rescale=1/255) for i in range(3)]
    arr_ite_datagen = []

    for i,obj_datagen in enumerate(arr_obj_datagen):
        print(f"Dataset: {arr_str_data_dir[i]}")
        it_datagen = obj_datagen.flow_from_directory(
                                                       None,
                                                        target_size=None,#as imagens sempre serão 150x150
                                                        batch_size=param_training.int_batch_size,#batch size definido pelo paramero
                                                        class_mode=None,
                                                        seed=Constantes.SEED
                                                        )
        arr_ite_datagen.append(it_datagen)
    return arr_ite_datagen

def fully_connected_model():
    #entrada
    entrada = Input(shape=(150,150,3),name="Entrada")

    #camadas a serem usadas
    achatar = None
    camada_um = None
    camada_dois = None
    camada_tres = None

    #camada de saida
    #lembre-se que é uma classificação binária
    saida = None

    #cria-se o modelo
    modelo = None
    return modelo


def simple_cnn_model(add_dropout=False):
    #entrada
    entrada = None

    #demais camadas









    achatar = layers.Flatten()(max_polling_d)
    if(add_dropout):
        achatar = None
    fc_a = None
    #camada de saida com 3 neuronios - cada um, respresntando uma classe
    #lembre de passar a cmada correta como saida
    #lembre-se que é uma classificação binária
    saida = None

    #cria-se o modelo
    modelo = Model(inputs=entrada, outputs=saida)
    return modelo


def run_model(model,it_gen_train,it_gen_validation,param_training,
                str_file_to_save,int_val_steps,
                load_if_exists=True):
    """
     model: Modelo criada
     it_gen_train: iterador do treino (usando o vetor gerado por meio da função `get_dataset`)
     it_gen_validation: iterador da validação (usando o vetor gerado por meio da função `get_dataset`)
     str_file_to_save: nome do arquivo em que o modelo é salvo
     param_training: (hiper)parametros do treino - objeto da classe ParametrosRedeNeural.
     int_val_steps: A validação também é feita em mini-batches (mesmo tamanho do treino).
     Especificar a quantidade de passos para iterar por todas as imagens de validação
     optmizer: objeto que representa o método de otimização que será usado (RMSProp, adam, por exemplo)
     load_if_exists: apenas carrega o modelo se ele já estiver salvo
    """
    if not load_if_exists or not os.path.isfile(str_file_to_save) :
        #ao compilar use o optimizador em param_training.optimizer a perda é uma entropia cruzada binária
        #a métrica será sempre acurácia
        model.compile(optimizer=None,loss=None,metrics=[None])
        #use o param_training para os parametros steps_per_epoch e epochs
        history = model.fit_generator(None,
                                            steps_per_epoch=None,
                                            epochs=None,
                                            #podemos colocar a validação e ver a validação por passos. Não recomento, pois, isso demoraria muio
                                            #..isso é bom apenas para analisarmos a curva de erro na validação e do treino. Mas, prefiro primeiramente
                                            #..analisar o resultado da validação apenas no final do treino - usando predict_generator - e, se necessário,
                                            #..habilitar essas linhas para um resultado mais detalhado
                                            #validation_data=it_gen_validation,
                                            #validation_steps=int_val_steps
                                         )
        #salve o modelo
        None
    else:
        #carrega o modelo
        model = None
    print("Avaliando validação....")
    loss, acc = model.evaluate_generator(None,
                                            steps=None)
    return acc
