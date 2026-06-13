from src.train import main
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

if __name__ == "__main__":
    main()
