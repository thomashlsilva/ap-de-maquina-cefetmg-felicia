from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import array_to_img
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from keras import Model
from keras import layers
from keras import Input
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
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

def get_dataset(param_training, arr_str_data_dir):
    """
    Cria três iteradores de imagens: treino, validação e teste.
    Cada um é um objeto iterável que gera mini-batches de imagens.

    Args:
        param_training: Objeto da classe ParametrosRedeNeural
        arr_str_data_dir: Lista com os caminhos para treino, validação e teste (nessa ordem)

    Returns:
        Lista com 3 iteradores (treino, validação e teste)
    """
    arr_obj_datagen = [ImageDataGenerator(rescale=1./255) for _ in range(3)]
    arr_ite_datagen = []

    for i, obj_datagen in enumerate(arr_obj_datagen):
        print(f"Dataset: {arr_str_data_dir[i]}")
        it_datagen = obj_datagen.flow_from_directory(
            arr_str_data_dir[i],
            target_size=(150, 150),  # as imagens sempre serão redimensionadas para 150x150
            batch_size=param_training.int_batch_size,
            class_mode='binary',     # porque é uma classificação binária: gato ou cachorro
            seed=Constantes.SEED
        )
        arr_ite_datagen.append(it_datagen)

    return arr_ite_datagen


def fully_connected_model():
    """
    Cria uma Rede Neural Totalmente Conectada (FC) para classificação de imagens 150x150x3.

    Retorna:
        modelo: Modelo Keras criado
    """
    # Entrada: imagens 150x150x3
    entrada = Input(shape=(150, 150, 3), name="Entrada")

    # Flatten (achatar a imagem)
    achatar = layers.Flatten()(entrada)

    # Primeira camada densa
    camada_um = layers.Dense(512, activation="relu", name="Camada1")(achatar)

    # Segunda camada densa
    camada_dois = layers.Dense(512, activation="relu", name="Camada2")(camada_um)

    # Camada de saída (classificação binária)
    saida = layers.Dense(1, activation="sigmoid", name="saida")(camada_dois)

    # Criar o modelo
    modelo = Model(inputs=entrada, outputs=saida)

    return modelo


def simple_cnn_model(add_dropout=False):
    """
    Cria uma CNN simples para classificação de imagens 150x150x3.

    Args:
        add_dropout (bool): Se True, adiciona camada de Dropout após Flatten.

    Retorna:
        modelo: Modelo Keras criado
    """
    entrada = Input(shape=(150, 150, 3), name="Entrada")

    # Primeira camada convolucional
    x = layers.Conv2D(32, (3, 3), activation="relu", name="Convolucao1")(entrada)
    x = layers.MaxPooling2D((2, 2))(x)  # Max Pooling

    # Segunda camada convolucional
    x = layers.Conv2D(64, (3, 3), activation="relu", name="Convolucao2")(x)
    x = layers.MaxPooling2D((2, 2))(x)  # Max Pooling

    # Terceira camada convolucional
    x = layers.Conv2D(128, (3, 3), activation="relu", name="Convolucao3")(x)
    x = layers.MaxPooling2D((2, 2))(x)  # Max Pooling

    # Quarta camada convolucional
    x = layers.Conv2D(128, (3, 3), activation="relu", name="Convolucao4")(x)
    x = layers.MaxPooling2D((2, 2))(x)  # Max Pooling

    # Achatar a imagem resultante
    x = layers.Flatten()(x)

    # Se a opção for ativada, adiciona dropout
    if add_dropout:
        x = layers.Dropout(0.5)(x)

    # Camada totalmente conectada (Fully Connected Layer)
    x = layers.Dense(512, activation="relu", name="CamadaFC")(x)

    # Camada de saída
    saida = layers.Dense(1, activation="sigmoid", name="saida")(x)

    modelo = Model(inputs=entrada, outputs=saida)
    return modelo



def run_model(model, train_generator, val_generator, epochs, model_save_path, steps_per_epoch, callbacks=None):
    """
    Função para compilar, treinar e avaliar um modelo Keras.
    
    Parâmetros:
    - model: O modelo Keras que será treinado.
    - train_generator: O gerador de dados para o treinamento.
    - val_generator: O gerador de dados para validação.
    - epochs: O número de épocas para o treinamento.
    - model_save_path: O caminho para salvar o modelo após o treinamento.
    - steps_per_epoch: O número de passos por época (normalmente o tamanho do dataset dividido pelo batch_size).
    """

    # 1. Compilação do modelo
    model.compile(
        optimizer=Adam(),  # Usando o otimizador Adam
        loss='binary_crossentropy',  # Função de perda para classificação binária
        metrics=['accuracy']  # Métrica de avaliação (acurácia)
    )

    # 2. Treinar o modelo usando fit
    history = model.fit(
        train_generator,         # Gerador de treinamento
        epochs=epochs,           # Número de épocas
        steps_per_epoch=steps_per_epoch,  # Passos por época
        validation_data=val_generator,  # Conjunto de validação
        validation_steps=steps_per_epoch,  # Passos de validação
        callbacks=callbacks  # Passando as callbacks, incluindo EarlyStopping
    )

    # 3. Salvar o modelo final
    model.save(model_save_path)  # Salvando o modelo final após o treinamento

    # 4. Avaliar o modelo no conjunto de validação
    val_loss, val_accuracy = model.evaluate(val_generator, steps=steps_per_epoch)
    print(f"Validação - Perda: {val_loss}, Acurácia: {val_accuracy}")
    
    return val_accuracy
