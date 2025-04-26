# ATUALIZADO para Keras 3.x / TensorFlow 2.x
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
from tensorflow.keras import Model, layers, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import math
import os.path

class Constantes():
    ARR_STR_DATA_DIR = ["data/treino",
                        "data/validacao",
                        "data/teste"]
    
    IDX_TREINO = 0
    IDX_VALIDACAO = 1
    IDX_TESTE = 2
    
    SEED = 2
    QTD_TREINO = 2048
    QTD_VALIDACAO = 1024
    QTD_TESTE = 1024

class ParametrosRedeNeural():
    def __init__(self, int_batch_size=64,
                 int_num_steps_per_epoch=None,
                 int_num_epochs=32,
                 optimizer=None):
        self.int_batch_size = int_batch_size
        self.int_num_epochs = int_num_epochs

        if not int_num_steps_per_epoch:
            self.int_num_steps_per_epoch = math.ceil(Constantes.QTD_TREINO/int_batch_size)
        else:
            self.int_num_steps_per_epoch = int_num_steps_per_epoch

        if not optimizer:
            self.optimizer = RMSprop(learning_rate=0.001, rho=0.9)
        else:
            self.optimizer = optimizer

def plot_imgs_from_iterator(it_datagen, num_lines, num_cols):
    i = 0
    bolFirst = True
    plt.figure(figsize=(9, 2*num_lines))
    for mat_x, arr_y in it_datagen:
        if bolFirst:
            print(f'x treino shape: {mat_x.shape}')
            print(f'y treino shape: {arr_y.shape}')
            bolFirst = False

        for idx_img in range(mat_x.shape[0]):
            plt.subplot(num_lines, num_cols, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(array_to_img(mat_x[idx_img]))
            plt.xlabel(f"Classe: {arr_y[idx_img]}")
            i += 1
        if i > (num_lines*num_cols)-1:
            break
    plt.show()

def get_dataset(param_training, arr_str_data_dir):
    arr_obj_datagen = [ImageDataGenerator(rescale=1/255) for _ in range(3)]
    arr_ite_datagen = []

    for i, obj_datagen in enumerate(arr_obj_datagen):
        print(f"Dataset: {arr_str_data_dir[i]}")
        it_datagen = obj_datagen.flow_from_directory(
            arr_str_data_dir[i],
            target_size=(150, 150),
            batch_size=param_training.int_batch_size,
            class_mode='binary',
            seed=Constantes.SEED
        )
        arr_ite_datagen.append(it_datagen)
    return arr_ite_datagen

def fully_connected_model():
    entrada = Input(shape=(150, 150, 3), name="Entrada")
    achatar = layers.Flatten()(entrada)
    camada_um = layers.Dense(512, activation='relu')(achatar)
    camada_dois = layers.Dense(256, activation='relu')(camada_um)
    camada_tres = layers.Dense(128, activation='relu')(camada_dois)
    saida = layers.Dense(1, activation='sigmoid')(camada_tres)
    
    modelo = Model(inputs=entrada, outputs=saida)
    return modelo

def simple_cnn_model(add_dropout=False):
    entrada = Input(shape=(150, 150, 3))
    
    # Bloco 1
    conv_1 = layers.Conv2D(32, (3, 3), activation='relu')(entrada)
    max_pool_1 = layers.MaxPooling2D((2, 2))(conv_1)
    
    # Bloco 2
    conv_2 = layers.Conv2D(64, (3, 3), activation='relu')(max_pool_1)
    max_pool_2 = layers.MaxPooling2D((2, 2))(conv_2)
    
    # Bloco 3
    conv_3 = layers.Conv2D(128, (3, 3), activation='relu')(max_pool_2)
    max_pool_3 = layers.MaxPooling2D((2, 2))(conv_3)
    
    # Bloco 4
    conv_4 = layers.Conv2D(128, (3, 3), activation='relu')(max_pool_3)
    max_pool_4 = layers.MaxPooling2D((2, 2))(conv_4)
    
    achatar = layers.Flatten()(max_pool_4)
    if add_dropout:
        achatar = layers.Dropout(0.5)(achatar)
    
    fc_a = layers.Dense(512, activation='relu')(achatar)
    saida = layers.Dense(1, activation='sigmoid')(fc_a)
    
    modelo = Model(inputs=entrada, outputs=saida)
    return modelo

def run_model(model, it_gen_train, it_gen_validation, param_training,
             str_file_to_save, int_val_steps, load_if_exists=True):
    
    if not load_if_exists or not os.path.isfile(str_file_to_save):
        model.compile(optimizer=param_training.optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        
        history = model.fit(
            it_gen_train,
            steps_per_epoch=param_training.int_num_steps_per_epoch,
            epochs=param_training.int_num_epochs
        )
        model.save(str_file_to_save)
    else:
        model = load_model(str_file_to_save)
    
    print("Avaliando validação....")
    loss, acc = model.evaluate(it_gen_validation, steps=int_val_steps)
    return acc