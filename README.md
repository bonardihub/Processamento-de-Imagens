
# Processamento de Imagens

## Autor

- Mateus Luz Francischini Bonardi


## Apresentação

Link para o vídeo sobre a apresentação do projeto final.

- https://drive.google.com/file/d/1X3VS2gdZrIbpc1CB3jfUSHl19_8eFOJo/view?usp=sharing
## Instalação

Para instalar e executar o projeto:

- Baixe [Python](https://www.python.org/downloads/release/python-3124/) na sua máquina, caso não tenha ainda.

- Certifique-se de baixar o código fonte do projeto no repositório.

- Abrir o projeto em sua IDE de preferência, [Visual Studio Code](https://code.visualstudio.com/download) é recomendado.

Navegue até a pasta principal do projeto em seu terminal.

```bash
  cd Processamento-de-Imagens-master
```

Instale as dependências
```bash
  pip install -r requirements.txt
``` 

Pronto! O projeto está preparado para ser executado.

**Importante:** As imagens do dataset devem ser inseridas na pasta "images_full", para posteriormente serem separadas através do "dataSplitter", resultando nas saidas "images_split".

## Descrição dos descritores implementados

- HOG (Histogram of Oriented Gradients)

Extrai características baseadas na distribuição dos gradientes de intensidade dentro de uma imagem. Ele divide a imagem em pequenas regiões, calcula um histograma de direções de gradiente, e depois normaliza esses histogramas.

- LBP (Local Binary Patterns)

Extrai características texturais convertendo a imagem em um conjunto de padrões binários locais. Ele compara cada pixel com seus vizinhos e codifica o padrão de diferença de intensidade em valores binários. A soma desses valores resulta em um histograma que representa a textura da imagem.

- Gabor Filter

Extrai características que são sensíveis às frequências espaciais e orientações específicas na imagem. Ele aplica filtros de Gabor, que são convoluções da imagem com funções seno/cosseno moduladas por uma gaussiana, para capturar informações de borda, textura e orientação em diferentes escalas e direções.

- Dense SIFT (Scale-Invariant Feature Transform)

Extrai características robustas e invariantes à escala e rotação de pontos de interesse na imagem. Ao contrário do SIFT tradicional, que detecta pontos de interesse esparsos, o Dense SIFT calcula descritores SIFT em uma grade densa, cobrindo uniformemente toda a imagem.


## Repositório do projeto

https://github.com/bonardihub/Processamento-de-Imagens
## Classificadores e Acurácia

### MLP

- HOG - 96.42%
- LBP - 96.64%
- Gabor Filter - 50%
- Dense SIFT - 100%

### SVM

- HOG - 100%
- LBP - 80.35%
- Gabor Filter - 92.85%
- Dense SIFT - 100%

### RF

- HOG - 100%
- LBP - 96.42%
- Gabor Filter - 92.85%
- Dense SIFT - 100%
