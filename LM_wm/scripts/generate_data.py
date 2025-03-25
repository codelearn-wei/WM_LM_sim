from datasets.data_generator import TrainingDataGenerator
from configs.config import MAP_PATH, RAW_DATA_PATH, TRAINING_DATA_DIR

def generate_training_data():
    """
    生成训练数据
    """
    print("开始生成训练数据...")
    data_generator = TrainingDataGenerator(MAP_PATH, RAW_DATA_PATH)
    data_generator.generate_training_data(TRAINING_DATA_DIR)
    print("训练数据生成完成！")

if __name__ == "__main__":
    generate_training_data() 