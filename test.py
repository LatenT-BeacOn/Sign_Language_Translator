
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tkinter as tk
from PIL import Image, ImageTk


# =========================================================
# SETTINGS
# =========================================================

WIDTH = 1300
HEIGHT = 750

offset = 20
imgSize = 300

labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N","O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X", "Y", "Z"
]


# =========================================================
# CAMERA + MODEL
# =========================================================

cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)

classifier = Classifier(
    "keras_model.h5",
    "labels.txt"
)


# =========================================================
# MAIN APPLICATION
# =========================================================

class SignLanguageApp:

    def __init__(self, root):

        self.root = root

        self.root.title("Sign Language To Text Conversion")
        self.root.geometry("1300x750")
        self.root.configure(bg="white")

        self.sentence = ""
        self.current_letter = ""

        # -------------------------------------------------
        # TITLE
        # -------------------------------------------------

        title = tk.Label(
            root,
            text="SIGN LANGUAGE TO TEXT CONVERSION",
            font=("Arial", 25, "bold"),
            bg="white"
        )

        title.pack(pady=15)


        # -------------------------------------------------
        # CAMERA FRAME
        # -------------------------------------------------

        self.camera_label = tk.Label(
            root,
            bg="black",
            width=650,
            height=500
        )

        self.camera_label.place(
            x=30,
            y=80,
            width=600,
            height=500
        )


        # -------------------------------------------------
        # PROCESSED HAND IMAGE
        # -------------------------------------------------

        self.processed_label = tk.Label(
            root,
            bg="gray"
        )

        self.processed_label.place(
            x=680,
            y=80,
            width=300,
            height=300
        )


        # -------------------------------------------------
        # PREDICTED CHARACTER
        # -------------------------------------------------

        tk.Label(
            root,
            text="Predicted Character",
            font=("Arial", 18, "bold"),
            bg="white"
        ).place(x=1000, y=100)

        self.character_label = tk.Label(
            root,
            text="-",
            font=("Arial", 45, "bold"),
            bg="white"
        )

        self.character_label.place(x=1070, y=145)


        # -------------------------------------------------
        # SENTENCE
        # -------------------------------------------------

        tk.Label(
            root,
            text="Sentence",
            font=("Arial", 18, "bold"),
            bg="white"
        ).place(x=680, y=410)

        self.sentence_label = tk.Label(
            root,
            text="",
            font=("Arial", 20),
            bg="white",
            wraplength=550,
            anchor="w",
            justify="left"
        )

        self.sentence_label.place(
            x=680,
            y=450,
            width=550,
            height=80
        )


        # -------------------------------------------------
        # BUTTON FRAME
        # -------------------------------------------------

        button_frame = tk.Frame(
            root,
            bg="white"
        )

        button_frame.place(
            x=680,
            y=560
        )


        # ADD CHARACTER BUTTON

        tk.Button(
            button_frame,
            text="Add Character",
            font=("Arial", 14, "bold"),
            command=self.add_character
        ).grid(row=0, column=0, padx=10)


        # SPACE BUTTON

        tk.Button(
            button_frame,
            text="Space",
            font=("Arial", 14, "bold"),
            command=self.add_space
        ).grid(row=0, column=1, padx=10)


        # DELETE BUTTON

        tk.Button(
            button_frame,
            text="Delete",
            font=("Arial", 14, "bold"),
            command=self.delete_character
        ).grid(row=0, column=2, padx=10)


        # CLEAR BUTTON

        tk.Button(
            button_frame,
            text="Clear",
            font=("Arial", 14, "bold"),
            command=self.clear_sentence
        ).grid(row=0, column=3, padx=10)


        # -------------------------------------------------
        # START VIDEO LOOP
        # -------------------------------------------------

        self.video_loop()


    # =====================================================
    # ADD CHARACTER
    # =====================================================

    def add_character(self):

        if self.current_letter not in ["", "nothing", "del", "space"]:

            self.sentence += self.current_letter

            self.sentence_label.config(
                text=self.sentence
            )


    # =====================================================
    # ADD SPACE
    # =====================================================

    def add_space(self):

        self.sentence += " "

        self.sentence_label.config(
            text=self.sentence
        )


    # =====================================================
    # DELETE CHARACTER
    # =====================================================

    def delete_character(self):

        if len(self.sentence) > 0:

            self.sentence = self.sentence[:-1]

            self.sentence_label.config(
                text=self.sentence
            )


    # =====================================================
    # CLEAR SENTENCE
    # =====================================================

    def clear_sentence(self):

        self.sentence = ""

        self.sentence_label.config(
            text=""
        )


    # =====================================================
    # VIDEO LOOP
    # =====================================================

    def video_loop(self):

        success, img = cap.read()

        if not success:

            self.root.after(
                10,
                self.video_loop
            )

            return


        # -------------------------------------------------
        # FLIP CAMERA
        # -------------------------------------------------

        img = cv2.flip(img, 1)

        imgOutput = img.copy()


        # -------------------------------------------------
        # HAND DETECTION
        # -------------------------------------------------

        hands, img = detector.findHands(img)


        # -------------------------------------------------
        # IF HAND FOUND
        # -------------------------------------------------

        if hands:

            hand = hands[0]

            x, y, w, h = hand['bbox']


            # -------------------------------------------------
            # CLAMP CROP REGION
            # -------------------------------------------------

            y1 = max(
                0,
                y - offset
            )

            y2 = min(
                img.shape[0],
                y + h + offset
            )

            x1 = max(
                0,
                x - offset
            )

            x2 = min(
                img.shape[1],
                x + w + offset
            )


            imgCrop = img[
                y1:y2,
                x1:x2
            ]


            # -------------------------------------------------
            # CHECK EMPTY CROP
            # -------------------------------------------------

            if imgCrop.size != 0:

                # ---------------------------------------------
                # WHITE CANVAS
                # ---------------------------------------------

                imgWhite = np.ones(
                    (imgSize, imgSize, 3),
                    np.uint8
                ) * 255


                # ---------------------------------------------
                # ASPECT RATIO
                # ---------------------------------------------

                aspectRatio = h / w


                if aspectRatio > 1:

                    k = imgSize / h

                    wCal = math.ceil(
                        k * w
                    )

                    imgResize = cv2.resize(
                        imgCrop,
                        (wCal, imgSize)
                    )

                    wGap = math.ceil(
                        (imgSize - wCal) / 2
                    )

                    imgWhite[
                        :,
                        wGap:wCal + wGap
                    ] = imgResize


                else:

                    k = imgSize / w

                    hCal = math.ceil(
                        k * h
                    )

                    imgResize = cv2.resize(
                        imgCrop,
                        (imgSize, hCal)
                    )

                    hGap = math.ceil(
                        (imgSize - hCal) / 2
                    )

                    imgWhite[
                        hGap:hCal + hGap,
                        :
                    ] = imgResize


                # ---------------------------------------------
                # CNN PREDICTION
                # ---------------------------------------------

                prediction, index = classifier.getPrediction(
                    imgWhite,
                    draw=False
                )


                # ---------------------------------------------
                # CURRENT LETTER
                # ---------------------------------------------

                self.current_letter = labels[index]


                self.character_label.config(
                    text=self.current_letter
                )


                # ---------------------------------------------
                # DRAW PREDICTION ON CAMERA
                # ---------------------------------------------

                cv2.rectangle(
                    imgOutput,
                    (x - offset, y - offset - 70),
                    (x - offset + 400, y - offset + 10),
                    (0, 255, 0),
                    cv2.FILLED
                )


                cv2.putText(
                    imgOutput,
                    self.current_letter,
                    (x, y - 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    2,
                    (0, 0, 0),
                    2
                )


                # ---------------------------------------------
                # DRAW BOUNDING BOX
                # ---------------------------------------------

                cv2.rectangle(
                    imgOutput,
                    (x - offset, y - offset),
                    (x + w + offset, y + h + offset),
                    (0, 255, 0),
                    4
                )


                # ---------------------------------------------
                # DISPLAY PROCESSED IMAGE
                # ---------------------------------------------

                processed_rgb = cv2.cvtColor(
                    imgWhite,
                    cv2.COLOR_BGR2RGB
                )

                processed_pil = Image.fromarray(
                    processed_rgb
                )

                processed_pil = processed_pil.resize(
                    (300, 300)
                )

                processed_imgtk = ImageTk.PhotoImage(
                    processed_pil
                )

                self.processed_label.imgtk = processed_imgtk

                self.processed_label.config(
                    image=processed_imgtk
                )


        else:

            self.character_label.config(
                text="-"
            )

            self.current_letter = ""


        # =================================================
        # DISPLAY CAMERA IN TKINTER
        # =================================================

        img_rgb = cv2.cvtColor(
            imgOutput,
            cv2.COLOR_BGR2RGB
        )

        img_pil = Image.fromarray(
            img_rgb
        )

        img_pil = img_pil.resize(
            (600, 500)
        )

        img_tk = ImageTk.PhotoImage(
            img_pil
        )

        self.camera_label.imgtk = img_tk

        self.camera_label.config(
            image=img_tk
        )


        # -------------------------------------------------
        # REPEAT LOOP
        # -------------------------------------------------

        self.root.after(
            10,
            self.video_loop
        )


# =========================================================
# CLOSE APPLICATION
# =========================================================

def close_application():

    cap.release()

    cv2.destroyAllWindows()

    root.destroy()


# =========================================================
# START APPLICATION
# =========================================================

root = tk.Tk()

app = SignLanguageApp(root)

root.protocol(
    "WM_DELETE_WINDOW",
    close_application
)

root.mainloop()
