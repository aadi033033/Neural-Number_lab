import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
import time


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Neural Number Lab",
    page_icon="∑",
    layout="wide"
)


# ==========================================
# CSS DESIGN
# ==========================================

st.markdown(
"""
<style>

.stApp {

    background:
    radial-gradient(circle at top, #101820, #000000);

    color:white;

}


/* Title */

.title {

    text-align:center;

    font-size:55px;

    font-weight:bold;

    color:#00ffff;

    text-shadow:
    0 0 15px cyan,
    0 0 30px cyan;

}



.subtitle {

    text-align:center;

    font-size:22px;

    color:#dddddd;

}



/* Instruction box */

.instruction {

    text-align:center;

    font-size:25px;

    padding:15px;

    border-radius:20px;

    background:
    rgba(0,255,255,0.08);

    border:1px solid cyan;

    box-shadow:
    0 0 15px cyan;

}



/* Prediction Cards */

.card {

    background:

    linear-gradient(
    145deg,
    rgba(0,255,255,0.15),
    rgba(255,255,255,0.05)
    );


    border-radius:25px;

    padding:25px;

    border:1px solid rgba(0,255,255,0.5);

    transition:0.4s;

}



.card:hover {

    transform:translateY(-12px);

    box-shadow:
    0 0 35px cyan;

}



.number {

    text-align:center;

    font-size:90px;

    color:#00ff99;

    font-weight:bold;

    text-shadow:
    0 0 20px #00ff99;

}



.conf {

    text-align:center;

    font-size:22px;

}



/* Button */

.stButton button {

    width:100%;

    height:55px;

    border-radius:30px;

    background:

    linear-gradient(
    90deg,
    #00ffff,
    #0066ff
    );


    color:white;

    font-size:22px;

    border:none;

}



.stButton button:hover {

    box-shadow:
    0 0 30px cyan;

    transform:scale(1.05);

}



/* Remove canvas extra black feeling */

canvas {

    border-radius:20px;

}



/* Thinking animation */

.thinking {

    text-align:center;

    font-size:30px;

    color:#00ffff;

    animation:pulse 1s infinite;

}



@keyframes pulse {

    0% {
        opacity:0.3;
    }

    50% {
        opacity:1;
    }

    100% {
        opacity:0.3;
    }

}


</style>

""",
unsafe_allow_html=True
)



# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():

    perceptron = load_model(
        "models/perceptron.keras",
        compile=False
    )


    ann = load_model(
        "models/ann.keras",
        compile=False
    )


    cnn = load_model(
        "models/cnn.keras",
        compile=False
    )


    return perceptron, ann, cnn



perceptron, ann, cnn = load_models()



# ==========================================
# TITLE
# ==========================================

st.markdown(
"""
<div class="title">

∑ Neural Number Lab ∑

</div>


<div class="subtitle">

Artificial Intelligence Mathematical Digit Recognition

<br>

f(x) = Neural Network → Number Prediction

</div>

""",
unsafe_allow_html=True
)



st.write("")



# ==========================================
# MAIN INSTRUCTION
# ==========================================

st.markdown(
"""
<div class="instruction">

✍️ Draw any number between 0 and 9

</div>

""",
unsafe_allow_html=True
)



st.write("")



# ==========================================
# CANVAS CENTERED
# ==========================================


left, center, right = st.columns(
    [1,2,1]
)


with center:

    canvas_result = st_canvas(

        fill_color="black",

        background_color="white",

        stroke_color="black",

        stroke_width=8,

        height=280,

        width=280,

        drawing_mode="freedraw",

        key="canvas"

    )



# ==========================================
# PREPROCESSING
# ==========================================

def preprocess(image):


    image = image[:, :, :3]


    img = Image.fromarray(
        image.astype("uint8")
    ).convert("L")


    img = np.array(img)



    # Invert because canvas is black digit on white

    img = 255 - img



    coords = np.argwhere(img > 20)



    if coords.size > 0:

        y0,x0 = coords.min(axis=0)

        y1,x1 = coords.max(axis=0)


        img = img[
            y0:y1+1,
            x0:x1+1
        ]



    img = Image.fromarray(img)


    img.thumbnail(
        (20,20)
    )


    img = np.array(img)



    final = np.zeros(
        (28,28),
        dtype=np.uint8
    )


    h,w = img.shape


    x = (28-w)//2

    y = (28-h)//2



    final[
        y:y+h,
        x:x+w
    ] = img



    final = final.astype(
        "float32"
    ) / 255.0



    return (

        final.reshape(1,28,28),

        final.reshape(1,28,28,1),

        final

    )



# ==========================================
# PREDICTION
# ==========================================


if canvas_result.image_data is not None:


    st.write("")


    if st.button(
        "🚀 Start AI Recognition"
    ):


        # AI thinking animation

        placeholder = st.empty()


        for i in range(3):

            placeholder.markdown(
            f"""

            <div class="thinking">

            🧠 AI Thinking{"."*(i+1)}

            </div>

            """,
            unsafe_allow_html=True
            )

            time.sleep(0.5)



        placeholder.empty()



        ann_input, cnn_input, display_img = preprocess(
            canvas_result.image_data
        )



        st.image(
            display_img,
            width=120,
            caption="AI Processed Image"
        )



        # Predictions

        p = perceptron.predict(
            ann_input,
            verbose=0
        )[0]


        a = ann.predict(
            ann_input,
            verbose=0
        )[0]


        c = cnn.predict(
            cnn_input,
            verbose=0
        )[0]



        p_digit = np.argmax(p)

        a_digit = np.argmax(a)

        c_digit = np.argmax(c)



        p_conf = np.max(p)*100

        a_conf = np.max(a)*100

        c_conf = np.max(c)*100



        st.divider()



        st.subheader(
            "🧮 Neural Network Results"
        )



        def result_card(
            name,
            digit,
            confidence
        ):

            st.markdown(
            f"""

            <div class="card">

            <h3 style="text-align:center">

            {name}

            </h3>


            <div class="number">

            {digit}

            </div>


            <div class="conf">

            Confidence:

            <b>{confidence:.2f}%</b>

            </div>


            </div>

            """,

            unsafe_allow_html=True
            )



        col1,col2,col3 = st.columns(3)



        with col1:

            result_card(
                "Single Layer Network",
                p_digit,
                p_conf
            )


        with col2:

            result_card(
                "ANN",
                a_digit,
                a_conf
            )


        with col3:

            result_card(
                "CNN",
                c_digit,
                c_conf
            )



        scores = {

            "Single Layer Network":p_conf,

            "ANN":a_conf,

            "CNN":c_conf

        }


        best = max(
            scores,
            key=scores.get
        )


        st.success(

            f"🏆 Best AI Model : {best} "
            f"({scores[best]:.2f}% confidence)"

        )