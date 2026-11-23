from Malkov import settings
from Malkov.sequences import convert, convert_diagnostic

def main() -> None:
    """Convierte las secuencias AST de settings.SOURCE_PATH en OUTPUT_DIR/{train,val,test}.npz para la CNN."""
    for name, path in convert(settings.SOURCE_PATH, settings.OUTPUT_DIR).items():
        print(f"{name}: {path}")
    print(f"ir_plag: {convert_diagnostic(settings.IR_PLAG_SOURCE_PATH, f'{settings.OUTPUT_DIR}/ir_plag.npz')}")

if __name__ == "__main__":
    main()
