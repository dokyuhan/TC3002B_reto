# Configuración temporal del paso Markov; esta carpeta se borra cuando otra persona entregue la versión final.
from config import PROCESSED_DIR

SOURCE_PATH = "../dataset/preprocessed/conplag_v2_sequences.npz"   # conplag_v1 o conplag_v2
IR_PLAG_SOURCE_PATH = "../dataset/preprocessed/ir_plag_sequences.npz"   # solo split 'diagnostic'
OUTPUT_DIR = PROCESSED_DIR
