# Classificação de Imagens com CNN

## Descrição
Projeto de aprendizado de máquina focado em classificação de imagens (gatos vs. cachorros) utilizando redes convolucionais (CNNs) com Keras (TensorFlow backend).

## Funcionalidades
- ✅ Construção de modelo CNN com Keras Functional API.
- ✅ Ajuste e validação de hiperparâmetros (épocas, batch size, learning rate, dropout).
- ✅ Treinamento com monitoramento via TensorBoard.
- ✅ Transfer Learning usando a arquitetura VGG16.
- ✅ Análise crítica de predições e diagnóstico de erros.

## Como Usar
```python
# Importar bibliotecas principais
import tensorflow as tf
from tensorflow import keras
# 1. Definir o modelo CNN (ou usar VGG16 para Transfer Learning)
# 2. Treinar o modelo com imagens pré-processadas
history = model.fit(train_generator, validation_data=val_generator, epochs=5)
# 3. Avaliar e visualizar os resultados

```

## Instalação de Dependências
Certifique-se de ter o Python e os pacotes necessários instalados:
```bash
apt-get install python3 jupyter python3-pip
pip install -r requirements.txt
```
Para visualizar o TensorBoard:
```bash
tensorboard --logdir=logs/fit
```

Observação: Em tarefas mais pesadas (ex.: Transfer Learning), recomenda-se usar GPU ou o [Google Colab](https://colab.research.google.com/).

### Executando o Jupyter Notebook
```bash
jupyter notebook
```
Abra o notebook fornecido e execute as células conforme as instruções no arquivo.

---

## Créditos

Atividade desenvolvida para a disciplina de Machine Learning do CEFET-MG, Campus Nova Gameleira.

Material baseado nas aulas do professor Daniel Hasan Dalip.