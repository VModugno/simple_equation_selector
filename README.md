# simple_equation_selector


install env

mamba env create -f environment.yaml
conda activate simple_eq



model based on https://github.com/chaodreaming/Simple-LaTeX-OCR.git (tested on simple equation works quite well)


to avoid error with ONNX in utils_load.py in simple-latex-ocr package replace line 21 with this 

#EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
EP_list = ['CPUExecutionProvider']