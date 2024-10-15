from flask import Flask, request, jsonify
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from flask_cors import CORS  
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)  

print("Loading model...")
try:
    base_model = DenseNet121(include_top=False, weights=None, input_shape=(64, 64, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(200, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()  # Print model architecture for debugging

    MODEL_PATH = 'optimal_densenet.keras'
    if os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH)
        print("Model weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")
except Exception as e:
    print(f"Error loading the model: {e}")
    traceback.print_exc()

CLASS_NAMES = ['墨.png', '竟.png', '章.png', '隐.png', '隔.png', '隘.png', '隙.png', '障.png', '隧.png',
               '隶.png', '难.png', '雀.png', '雁.png', '雄.png', '雅.png', '集.png', '雇.png', '雌.png',
               '雍.png', '雏.png', '雕.png', '雨.png', '雪.png', '零.png', '雷.png', '雹.png', '雾.png',
               '需.png', '霄.png', '震.png', '霉.png', '霍.png', '霓.png', '霖.png', '霜.png', '霞.png',
               '露.png', '霸.png', '霹.png', '青.png', '靖.png', '静.png', '靛.png', '非.png', '靠.png',
               '靡.png', '面.png', '革.png', '靳.png', '靴.png', '靶.png', '鞋.png', '鞍.png', '鞘.png',
               '鞠.png', '鞭.png', '韦.png', '韧.png', '韩.png', '韭.png', '音.png', '韵.png', '韶.png',
               '页.png', '顶.png', '顷.png', '项.png', '顺.png', '须.png', '顽.png', '顾.png', '顿.png',
               '颁.png', '颂.png', '预.png', '颅.png', '领.png', '颇.png', '颈.png', '颊.png', '颐.png',
               '频.png', '颓.png', '颖.png', '颗.png', '题.png', '颜.png', '额.png', '颠.png', '颤.png',
               '颧.png', '风.png', '飘.png', '飞.png', '食.png', '餐.png', '饥.png', '饭.png', '饮.png',
               '饯.png', '饰.png', '饱.png', '饲.png', '饵.png', '饶.png', '饺.png', '饼.png', '饿.png',
               '馁.png', '馅.png', '馆.png', '馈.png', '馋.png', '馏.png', '馒.png', '首.png', '香.png',
               '马.png', '驭.png', '驮.png', '驯.png', '驰.png', '驱.png', '驳.png', '驴.png', '驶.png',
               '驹.png', '驻.png', '驼.png', '驾.png', '骂.png', '骄.png', '骆.png', '骇.png', '骋.png',
               '验.png', '骏.png', '骑.png', '骗.png', '骚.png', '骡.png', '骤.png', '骨.png', '骸.png',
               '髓.png', '高.png', '鬃.png', '鬼.png', '魁.png', '魂.png', '魄.png', '魏.png', '魔.png',
               '鱼.png', '鲁.png', '鲍.png', '鲜.png', '鲤.png', '鲸.png', '鳃.png', '鳖.png', '鳞.png',
               '鸟.png', '鸡.png', '鸣.png', '鸥.png', '鸦.png', '鸭.png', '鸯.png', '鸳.png', '鸵.png',
               '鸽.png', '鸿.png', '鹃.png', '鹅.png', '鹊.png', '鹏.png', '鹤.png', '鹰.png', '鹿.png',
               '麓.png', '麦.png', '麻.png', '黄.png', '黍.png', '黎.png', '黑.png', '黔.png', '默.png',
               '鼎.png', '鼓.png', '鼠.png', '鼻.png', '齐.png', '齿.png', '龄.png', '龋.png', '龙.png',
               '龚.png', '龟.png']

@app.route('/')
def home():
    return "<h1>Welcome to the Image Classification API</h1><p>Use the /predict endpoint to upload an image for classification.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((64, 64))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0 

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        return jsonify({'predicted_class': CLASS_NAMES[predicted_class]})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)