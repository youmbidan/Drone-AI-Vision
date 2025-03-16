import sys
from cProfile import label

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QPushButton,
    QGraphicsDropShadowEffect, QHBoxLayout, QStackedWidget, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem,
    QMessageBox, QFrame,
    QFileDialog, QProgressBar, QGridLayout
)
from PyQt6.QtGui import QFont, QMovie, QColor, QMovie, QPixmap, QPalette, QBrush, QLinearGradient, QImage, \
    QPainter, QCursor, QIcon  # Import QPainter
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QPushButton,
    QGraphicsDropShadowEffect, QHBoxLayout, QStackedWidget, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem,
    QMessageBox, QFrame, QGridLayout, QSpacerItem, QSizePolicy , QListWidget, QListWidgetItem)

from PyQt6.QtCore import Qt, QDir, QTimer, QRect, QPropertyAnimation, QEasingCurve, pyqtSignal, QThread, QMargins, QUrl, \
    QPoint, QPropertyAnimation, QSize
from PyQt6.QtWidgets import QGraphicsDropShadowEffect,QComboBox
import pyttsx3
import threading
import random
import cv2
import os  # Import the os module for directory operations
from datetime import datetime  # Import datetime for timestamping
import shutil  # Import shutil for file copying



from testeur import generate_report


class VideoGraphicsItem(QGraphicsPixmapItem):
    def __init__(self, parent=None):
        super().__init__(parent)

    def updateFrame(self, frame):
        # Convertir l'image OpenCV en QImage
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qImg)
        self.setPixmap(pixmap)


# ----- Écran de chargement -----



class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone AI - Chargement...")
        self.setGeometry(300, 200, 500, 300)
        self.setFixedSize(500, 300)
        self.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF; /* Fond blanc */
                color: #003366; /* Texte noir */
            }
        """)

        shadow_effect = QGraphicsDropShadowEffect()
        shadow_effect.setBlurRadius(15)  # Réduire le flou
        shadow_effect.setXOffset(1)
        shadow_effect.setYOffset(1)
        shadow_effect.setColor(QColor(0, 0, 0, 100))  # Ombre noire, moins intense

        self.loading_gif = QLabel(self)
        gif_path = QDir.currentPath() + "/Vidéos/load.gif"
        self.movie = QMovie(gif_path)
        self.loading_gif.setMovie(self.movie)
        self.loading_gif.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_gif.setGraphicsEffect(shadow_effect)
        self.movie.start()

        self.loading_label = QLabel("Chargement de Drone AI...", self)
        self.loading_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setStyleSheet("color: #003366;")  # Texte noir
        self.loading_label.setGraphicsEffect(shadow_effect)

        layout = QVBoxLayout(self)
        layout.addWidget(self.loading_gif)
        layout.addWidget(self.loading_label)
        self.setLayout(layout)


# ----- Page d'analyse -----

class AnalysePage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.engine = pyttsx3.init()
        self.current_step = 0
        self.main_window.home_page.language_combo.currentIndexChanged.connect(self.update_language) #connect to home page signal
        self.current_language = "fr"  # Initialiser la langue par défaut

        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("Erreur: Impossible d'ouvrir la caméra.")
            sys.exit()

        # Définit le dossier de capture prédéfini
        self.capture_folder_name = "captured_images"
        self.base_path = "C:/Users/Danielle/Desktop/stage_N3/"  # Chemin de base
        self.capture_folder = os.path.join(self.base_path, self.capture_folder_name)
        # Crée le dossier de capture s'il n'existe pas
        if not os.path.exists(self.capture_folder):
            try:
                os.makedirs(self.capture_folder)
                print(f"Dossier de capture créé : {self.capture_folder}")
            except OSError as e:
                print(f"Erreur lors de la création du dossier : {e}")
                self.capture_folder = None

        self.captured_images = []
        self.initUI()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        self.analysis_timer = QTimer(self)
        self.analysis_timer.timeout.connect(self.perform_step)

        self.motor_control_enabled = True  # Initially enable motor control

        self.rotation_delay = 3  # Delay for rotation in seconds
        self.capture_delay = 5  # Delay for capture in seconds
        self.rotation_angles = [360, 180, 360]  # Angles de rotation doublés

    def initUI(self):
        self.setWindowTitle(self.translate_text("Analyse de Drone Guidée"))
        self.setGeometry(200, 100, 1200, 600)
        self.setStyleSheet(
            "background-color: #FFFFFF; color: #00008B;"
        )  # Background en blanc, texte en bleu foncé

        # Partie "Caméra"
        self.camera_label = QLabel(self.translate_text("Vue Caméra"), self)
        self.camera_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("color: #800080;")  # Texte en violet

        self.camera_scene = QGraphicsScene()
        self.camera_view = QGraphicsView(self.camera_scene)
        self.camera_view.setStyleSheet(
            """
            background-color: #E6E6FA; /* Lavande pour la zone */
            border: 1px solid #00008B;  /* Bordure en bleu foncé */
            border-radius: 10px;
        """
        )

        self.video_item = VideoGraphicsItem()
        self.camera_scene.addItem(self.video_item)
        self.video_item.setPos(0, 0)

        # Partie "Illustrations"
        self.illustration_label = QLabel(self.translate_text("Illustrations"), self)
        self.illustration_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.illustration_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.illustration_label.setStyleSheet("color: #800080;")  # Texte en violet

        self.step_label = QLabel(self.translate_text("Étape 1 : Vue de dessus"), self)
        self.step_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.step_label.setStyleSheet("color: #800080;")  # Texte en violet

        self.instruction_label = QLabel(
            self.translate_text("Veuillez placer le drone de manière à ce qu'il soit vu de dessus."), self
        )
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instruction_label.setStyleSheet("color: #4B0082;")  # Texte en violet foncé

        self.image_scene = QGraphicsScene()
        self.image_view = QGraphicsView(self.image_scene)
        self.image_view.setStyleSheet(
            """
            background-color: #E6E6FA; /* Lavande pour la zone */
            border: 1px solid #00008B;   /* Bordure en bleu foncé */
            border-radius: 10px;
        """
        )
        self.showImage("images/Dessus.png")

        # Barre de Progression
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 4)  # Changed range to 0-4 for 5 steps (0-4 index)
        self.progress_bar.setValue(self.current_step)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid #00008B;  /* Bordure bleu foncé */
                border-radius: 5px;
                text-align: center;
                color: white;
                background-color: #4B0082;  /* Violet foncé */
            }

            QProgressBar::chunk {
                background-color: #800080;  /* Violet */
                border-radius: 5px;
            }
        """
        )

        # Boutons - Désactivés initialement
        self.capture_button = QPushButton(self.translate_text("Capturer l'image"), self)
        self.capture_button.setStyleSheet(
            """
            QPushButton {
                background-color: #E6E6FA;   /* Lavande */
                border: 2px solid #00008B;  /* Bordure bleu foncé */
                color: #4B0082;    /* Texte en violet foncé */
                padding: 8px 16px;
                text-align: center;
                text-decoration: none;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #D8BFD8; /* Lavande plus clair au survol */
            }
        """
        )
        self.capture_button.clicked.connect(self.captureImage)
        self.capture_button.setIcon(QIcon(QDir.currentPath() + "/images/camera.png"))
        self.capture_button.setEnabled(False)

        self.next_button = QPushButton(self.translate_text("Étape suivante"), self)
        self.next_button.setStyleSheet(
            """
            QPushButton {
                background-color: #E6E6FA;    /* Lavande */
                border: 2px solid #00008B;   /* Bordure bleu foncé */
                color: #4B0082;     /* Texte en violet foncé */
                padding: 8px 16px;
                text-align: center;
                text-decoration: none;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #D8BFD8;  /* Lavande plus clair au survol */
            }
        """
        )
        self.next_button.clicked.connect(self.nextStep)
        self.next_button.setIcon(QIcon(QDir.currentPath() + "/images/next.png"))
        self.next_button.setEnabled(False)

        # Add Folder Label with Information of the selected folder
        self.folder_info_label = QLabel(f"{self.translate_text('Capture Folder')}: {self.capture_folder}")
        self.folder_info_label.setStyleSheet("color: #00008B;")
        self.folder_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Layout horizontal principal
        hbox = QHBoxLayout(self)

        # Layout vertical pour la partie Caméra
        camera_vbox = QVBoxLayout()
        camera_vbox.addWidget(self.camera_label)
        camera_vbox.addWidget(self.camera_view)
        hbox.addLayout(camera_vbox)

        # Layout vertical pour la partie Illustrations
        illustration_vbox = QVBoxLayout()
        illustration_vbox.addWidget(self.illustration_label)
        illustration_vbox.addWidget(self.step_label)
        illustration_vbox.addWidget(self.instruction_label)
        illustration_vbox.addWidget(self.image_view)
        illustration_vbox.addWidget(self.progress_bar)
        illustration_vbox.addWidget(self.capture_button)
        illustration_vbox.addWidget(self.next_button)
        illustration_vbox.addWidget(self.folder_info_label)
        hbox.addLayout(illustration_vbox)

        self.setLayout(hbox)
        self.translate_ui()
        self.show()

        # Initially disable the motor
        self.send_serial_command("0")  # Send '0' to stop the motor on initialization
        QTimer.singleShot(500, lambda: self.speak(
            self.translate_text("Bienvenue sur Drone AI vision. Votre IA de detction des anomalies visuelles des drones. Appuyez sur le bouton demarrér pour commencer.")))

    def start_analysis_sequence(self):
        self.current_step = 0  # Reset the step
        self.update_progress_bar()  # Update progress bar after reset
        # self.analysis_timer.start(5000)  # Start the timer  removed it so the process will not continue automatically

        # Start with the first step immediately
        self.perform_step()
        self.motor_control_enabled = True  # Enable motor control at the start of the sequence

    def perform_step(self):
        if self.current_step == 0:
            self.step_label.setText(self.translate_text("Étape 1 : Vue de dessus"))
            self.instruction_label.setText(
                self.translate_text("Veuillez placer le drone de manière à ce qu'il soit vu de dessus.")
            )
            self.showImage("images/Dessus.png")
            self.speak(self.translate_text("Veuillez placer le drone de manière à ce qu'il soit vu de dessus."))
            QTimer.singleShot(self.capture_delay * 1000, self.captureImage)


        elif self.current_step == 1:
            self.step_label.setText(self.translate_text("Étape 2 : Vue de dessous"))
            self.instruction_label.setText(
                self.translate_text("Rotation de 180 degrés. Vue de dessous.")
            )
            self.showImage("images/Dessous.png")
            self.speak(self.translate_text("Rotation de 180 degrés. Vue de dessous."))

            self.rotate_and_capture(self.rotation_angles[0], self.capture_delay)

        elif self.current_step == 2:
            self.step_label.setText(self.translate_text("Étape 3 : Vue de gauche"))
            self.instruction_label.setText(
                self.translate_text("Rotation de 180 degrés. Vue de gauche.")
            )
            self.showImage("images/Gauche.png")
            self.speak(self.translate_text("Rotation de 180 degrés. Vue de gauche."))
            self.rotate_and_capture(self.rotation_angles[1], self.capture_delay)

        elif self.current_step == 3:
            self.step_label.setText(self.translate_text("Étape 4 : Vue de droite"))
            self.instruction_label.setText(
                self.translate_text("Rotation de 180 degrés. Vue de droite.")
            )
            self.showImage("images/Droite.png")
            self.speak(self.translate_text("Rotation de 180 degrés. Vue de droite."))
            self.rotate_and_capture(self.rotation_angles[2], self.capture_delay)

        elif self.current_step == 4:  # Added step 4 for processing
            # self.analysis_timer.stop()  removed
            self.step_label.setText(self.translate_text("Étape 5 : En cours d'analyse"))
            self.instruction_label.setText(self.translate_text("L'analyse du drone est en cours..."))
            self.showImage("images/processing.png") # Or a relevant image for processing
            print("Dans la partie ELSE - Lancement de l'analyse et arrêt du moteur...")
            self.speak(self.translate_text("Lancement de l'analyse du drone."))
            QMessageBox.information(self, self.translate_text("Analyse"), self.translate_text("L'analyse du drone a été lancée."))
            # Stop the motor at the end of the sequence
            if self.motor_control_enabled:
                self.stopMotor()

            self.main_window.show_processing_page(self.captured_images)
            return

        self.current_step += 1
        self.update_progress_bar()

    def rotate_and_capture(self, angle, capture_delay):
        if self.motor_control_enabled:
            self.send_serial_command(str(angle))
            QTimer.singleShot(self.rotation_delay * 1000, lambda: self.capture_after_rotation(capture_delay)) # added this to be more clear , the rotation_delay is for rotation time
        else:
            print("Motor control disabled, skipping rotation.")

    def capture_after_rotation(self, capture_delay):  # added this for capture after rotation
        QTimer.singleShot(capture_delay * 1000, self.captureImage)

    def capture_after_delay(self, delay):
        QTimer(self).singleShot(delay * 1000, self.captureImage)

    def send_serial_command(self, command):
        if self.main_window.confirmation_page.serial_connection and self.main_window.confirmation_page.serial_connection.is_open:
            try:
                print(f"Envoi de la commande série: {command}")
                self.main_window.confirmation_page.serial_connection.write(command.encode())
                print(f"Commande série envoyée: {command}")
            except serial.SerialException as e:
                print(f"Erreur lors de l'envoi des données: {e}")
            except Exception as e:
                print(f"Une erreur inattendue s'est produite: {e}")
        else:
            print("La connexion série n'est pas disponible.")

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            self.video_item.updateFrame(frame)

    def showImage(self, filename):
        image_path = QDir.currentPath() + "/" + filename
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.image_scene.clear()
            item = QGraphicsPixmapItem(pixmap.scaledToWidth(300))
            self.image_scene.addItem(item)
        else:
            print(f"Erreur: Impossible de charger l'image à {image_path}")

    def captureImage(self):
        if not self.capture_folder:
            QMessageBox.warning(
                self,
                self.translate_text("Attention"),
                self.translate_text("Le dossier de capture n'est pas défini. Veuillez vérifier le chemin d'accès."),
            )
            self.speak(self.translate_text("Le dossier de capture n'est pas défini."))
            print("Le dossier de capture n'est pas défini.")
            return

        ret, frame = self.camera.read()
        if not ret:
            QMessageBox.critical(
                self, self.translate_text("Erreur"), self.translate_text("Impossible de capturer l'image. Veuillez vérifier la caméra.")
            )
            return

        try:
            # Générer un nom de fichier unique avec un horodatage
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                self.capture_folder, f"step_{self.current_step + 1}_{timestamp}.png"
            )

            print(f"Tentative d'enregistrement de l'image : {filename}")

            # Enregistrer l'image capturée
            cv2.imwrite(filename, frame)
            self.captured_images.append(filename)

            if os.path.exists(filename):
                print(f"Image capturée et enregistrée sous: {filename}")
                self.speak(
                    self.translate_text("Image capturée. Veuillez passer à l'étape suivante."))
                QMessageBox.information(
                    self,
                    self.translate_text("Image Capturée"),
                    self.translate_text("Image capturée. Veuillez passer à l'étape suivante."),
                )
                self.perform_step()  # To continue to the next step after the photo captured
            else:
                QMessageBox.critical(
                    self,
                    self.translate_text("Erreur"),
                    self.translate_text(f"Impossible d'enregistrer l'image à {filename}. Vérifiez les permissions du dossier."),
                )
                print(
                    self.translate_text(f"Impossible d'enregistrer l'image à {filename}. Vérifiez les permissions du dossier.")
                )
        except Exception as e:
            QMessageBox.critical(
                self, self.translate_text("Erreur"), self.translate_text(f"Une erreur s'est produite lors de l'enregistrement de l'image : {e}")
            )
            print(f"Erreur lors de l'enregistrement de l'image : {e}")

    def nextStep(self):
        # This method is now deprecated as the analysis is automatic
        pass

    def stopMotor(self):
        if self.main_window.confirmation_page.serial_connection and self.main_window.confirmation_page.serial_connection and self.main_window.confirmation_page.serial_connection.is_open:
            print("About to send '0' to Arduino to stop motor...")
            try:
                self.main_window.confirmation_page.serial_connection.write(b"0")
                print("Successfully sent '0' to Arduino to stop motor.")
            except serial.SerialException as e:
                print(f"Error sending data: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
        else:
            print("Serial connection not available to stop motor.")

    def closeEvent(self, event):
        self.timer.stop()
        self.camera.release()
        cv2.destroyAllWindows()

    def update_progress_bar(self):
        """Mise à jour de la barre de progression."""
        progress = int((self.current_step / 4) * 100)  # Assuming 4 steps represent 100%
        self.progress_bar.setValue(self.current_step) # set value with steps to make sure correct progress bar

    def speak(self, text):
        threading.Thread(target=self._speak, args=(text,)).start()

    def _speak(self, text):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Erreur lors de la synthèse vocale : {e}")

    def translate_text(self, text):
        translations = {
            "Analyse de Drone Guidée": {"fr": "Analyse de Drone Guidée", "en": "Guided Drone Analysis", "es": "Análisis Guiado de Drones"},
            "Vue Caméra": {"fr": "Vue Caméra", "en": "Camera View", "es": "Vista de Cámara"},
            "Illustrations": {"fr": "Illustrations", "en": "Illustrations", "es": "Ilustraciones"},
            "Étape 1 : Vue de dessus": {"fr": "Étape 1 : Vue de dessus", "en": "Step 1: Top View", "es": "Paso 1: Vista Superior"},
            "Veuillez placer le drone de manière à ce qu'il soit vu de dessus.": {"fr": "Veuillez placer le drone de manière à ce qu'il soit vu de dessus.", "en": "Please position the drone so it is viewed from above.", "es": "Por favor, coloque el dron de manera que se vea desde arriba."},
            "Capturer l'image": {"fr": "Capturer l'image", "en": "Capture Image", "es": "Capturar Imagen"},
            "Étape suivante": {"fr": "Étape suivante", "en": "Next Step", "es": "Siguiente Paso"},
            "Capture Folder": {"fr": "Dossier de capture", "en": "Capture Folder", "es": "Carpeta de captura"},
            "Étape 2 : Vue de dessous": {"fr": "Étape 2 : Vue de dessous", "en": "Step 2: Bottom View", "es": "Paso 2: Vista Inferior"},
            "Rotation de 180 degrés. Vue de dessous.": {"fr": "Rotation de 180 degrés. Vue de dessous.", "en": "180 degree rotation. Bottom view.", "es": "Rotación de 180 grados. Vista inferior."},
            "Étape 3 : Vue de gauche": {"fr": "Étape 3 : Vue de gauche", "en": "Step 3: Left View", "es": "Paso 3: Vista Izquierda"},
            "Rotation de 180 degrés. Vue de gauche.": {"fr": "Rotation de 180 degrés. Vue de gauche.", "en": "180 degree rotation. Left view.", "es": "Rotación de 180 grados. Vista izquierda."},
            "Étape 4 : Vue de droite": {"fr": "Étape 4 : Vue de droite", "en": "Step 4: Right View", "es": "Paso 4: Vista Derecha"},
            "Rotation de 180 degrés. Vue de droite.": {"fr": "Rotation de 180 degrés. Vue de droite.", "en": "180 degree rotation. Right view.", "es": "Rotación de 180 grados. Vista derecha."},
            "Lancement de l'analyse du drone.": {"fr": "Lancement de l'analyse du drone.", "en": "Starting drone analysis.", "es": "Iniciando el análisis del dron."},
            "Analyse": {"fr": "Analyse", "en": "Analysis", "es": "Análisis"},
            "L'analyse du drone a été lancée.": {"fr": "L'analyse du drone a été lancée.", "en": "The drone analysis has been launched.", "es": "El análisis del dron ha sido lanzado."},
            "Bienvenue sur Drone AI vision. Votre IA de detction des anomalies visuelles des drones. Appuyez sur le bouton demarrér pour commencer.": {"fr": "Bienvenue sur Drone AI vision. Votre IA de detction des anomalies visuelles des drones. Appuyez sur le bouton demarrér pour commencer.", "en": "Welcome to Drone AI vision. Your AI for detecting visual anomalies in drones. Press the start button to begin.", "es": "Bienvenido a Drone AI vision. Su IA para la detección de anomalías visuales en drones. Presione el botón de inicio para comenzar."},
            "Attention": {"fr": "Attention", "en": "Attention", "es": "Atención"},
            "Le dossier de capture n'est pas défini. Veuillez vérifier le chemin d'accès.": {"fr": "Le dossier de capture n'est pas défini. Veuillez vérifier le chemin d'accès.", "en": "The capture folder is not defined. Please check the path.", "es": "La carpeta de captura no está definida. Por favor, verifique la ruta."},
            "Erreur": {"fr": "Erreur", "en": "Error", "es": "Error"},
            "Impossible de capturer l'image. Veuillez vérifier la caméra.": {"fr": "Impossible de capturer l'image. Veuillez vérifier la caméra.", "en": "Unable to capture image. Please check the camera.", "es": "No se puede capturar la imagen. Por favor, revise la cámara."},
            "Image capturée. Veuillez passer à l'étape suivante.": {"fr": "Image capturée. Veuillez passer à l'étape suivante.", "en": "Image captured. Please proceed to the next step.", "es": "Imagen capturada. Por favor, proceda al siguiente paso."},
            "Image Capturée": {"fr": "Image Capturée", "en": "Image Captured", "es": "Imagen Capturada"},
            "Impossible d'enregistrer l'image à {}. Vérifiez les permissions du dossier.": {"fr": "Impossible d'enregistrer l'image à {}. Vérifiez les permissions du dossier.", "en": "Unable to save the image to {}. Check folder permissions.", "es": "No se puede guardar la imagen en {}. Verifique los permisos de la carpeta."},
            "Une erreur s'est produite lors de l'enregistrement de l'image : {}": {"fr": "Une erreur s'est produite lors de l'enregistrement de l'image : {}", "en": "An error occurred while saving the image: {}", "es": "Se produjo un error al guardar la imagen: {}"}
        }
        return translations.get(text, {}).get(self.current_language, text)

    def translate_ui(self):
        self.setWindowTitle(self.translate_text("Analyse de Drone Guidée"))
        self.camera_label.setText(self.translate_text("Vue Caméra"))
        self.illustration_label.setText(self.translate_text("Illustrations"))
        self.step_label.setText(self.translate_text("Étape 1 : Vue de dessus"))
        self.instruction_label.setText(self.translate_text("Veuillez placer le drone de manière à ce qu'il soit vu de dessus."))
        self.capture_button.setText(self.translate_text("Capturer l'image"))
        self.next_button.setText(self.translate_text("Étape suivante"))
        self.folder_info_label.setText(f"{self.translate_text('Capture Folder')}: {self.capture_folder}")

    def update_language(self, index):
        """
        Méthode appelée lorsqu'une nouvelle langue est sélectionnée dans le QComboBox de HomePage.
        Met à jour la langue courante et traduit l'interface utilisateur.
        """
        self.current_language = self.main_window.home_page.language_combo.itemData(index)
        self.translate_ui()

    def update_language(self, index):
        """
        Méthode appelée lorsqu'une nouvelle langue est sélectionnée dans le QComboBox de HomePage.
        Met à jour la langue courante et traduit l'interface utilisateur.
        """
        self.current_language = self.main_window.home_page.language_combo.itemData(index)
        self.translate_ui()
        # Mise à jour dynamique des textes des étapes en fonction de la langue
        self.update_step_texts()

    def update_step_texts(self):
        """Met à jour les textes des étapes et instructions en fonction de la langue."""
        if self.current_step == 0:
            self.step_label.setText(self.translate_text("Étape 1 : Vue de dessus"))
            self.instruction_label.setText(
                self.translate_text("Veuillez placer le drone de manière à ce qu'il soit vu de dessus."))
        elif self.current_step == 1:
            self.step_label.setText(self.translate_text("Étape 2 : Vue de dessous"))
            self.instruction_label.setText(
                self.translate_text("Rotation de 180 degrés. Vue de dessous."))
        elif self.current_step == 2:
            self.step_label.setText(self.translate_text("Étape 3 : Vue de gauche"))
            self.instruction_label.setText(
                self.translate_text("Rotation de 180 degrés. Vue de gauche."))
        elif self.current_step == 3:
            self.step_label.setText(self.translate_text("Étape 4 : Vue de droite"))
            self.instruction_label.setText(
                self.translate_text("Rotation de 180 degrés. Vue de droite."))

# --- Page de Traitement ---
class IAThread:
    pass



class ImageBox(QWidget):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)

        self.image_path = image_path

        # Layout principal
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # Supprimer les marges

        # Label pour afficher l'image
        self.image_label = QLabel()
        pixmap = QPixmap(self.image_path).scaledToWidth(150, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Ajouter l'effet d'ombre portée
        shadow_effect = QGraphicsDropShadowEffect()
        shadow_effect.setBlurRadius(20)  # Adoucir les bords de l'ombre
        shadow_effect.setColor(QColor(75, 0, 130, 150))  # Couleur violet foncé (4B0082) et transparence
        shadow_effect.setOffset(10, 10)  # Décalage de l'ombre (x, y)
        self.setGraphicsEffect(shadow_effect)

        # Style du widget pour la boîte blanche
        self.setStyleSheet("background-color: white;")

        layout.addWidget(self.image_label)
        self.setLayout(layout)


class ProcessingThread(QThread):
    progress_update = pyqtSignal(int)  # Signal pour la mise à jour de la progression
    finished = pyqtSignal(str) # signal pour le chemin du rapport

    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths

    def run(self):
        # Simuler un traitement long
        num_images = len(self.image_paths)
        report_path = None  # Initialiser le chemin du rapport

        try:
            # Simuler un traitement long
            num_images = len(self.image_paths)
            for i, path in enumerate(self.image_paths):
                # Simuler un traitement
                QThread.msleep(500)  # Pause de 500 ms

                # Calculer la progression
                progress = int((i + 1) / num_images * 100)
                self.progress_update.emit(progress)

            # Simuler la création du rapport
            QThread.msleep(1000)  # Pause de 1 seconde

            # Simuler le chemin du rapport
            report_path = "C:/Users/Danielle/Desktop/stage_N3/Rapport/rapport.pdf"  # Remplacez par le chemin réel

            self.finished.emit(report_path)

        except Exception as e:
            print(f"Erreur lors du traitement: {e}")
            self.finished.emit(None)  # Émettre None en cas d'erreur


class ProcessingPage(QWidget):
    processing_finished = pyqtSignal(str)  # Signal pour le chemin du rapport

    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths  # Recevoir les chemins d'images
        self.initUI()

    def analysePhotos(self, captured_images, metadata):  # Accepter maintenant 2 paramètre
        self.ia_thread = IAThread(captured_images, metadata)  # Lancement du thread, avec les deux parametres.
        self.ia_thread.finished.connect(self.AnalyseDone)  # Connecte le signal à la fonction de récupération
        self.ia_thread.start()
    def initUI(self):
        self.setWindowTitle("Traitement des Images")
        self.setGeometry(200, 100, 800, 600)
        self.setStyleSheet("background-color: #FFFFFF; color: #00008B;") #Changement palette de couleur

        # Layout principal
        main_layout = QVBoxLayout(self)

        # Label titre
        title_label = QLabel("Traitement des Images", self)
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #800080;")
        main_layout.addWidget(title_label)

        # Créer et ajouter des QListWidget pour chaque image
        self.image_list = QListWidget()
        self.image_list.setFlow(QListWidget.Flow.LeftToRight) #Affichage horizontal
        self.image_list.setWrapping(False) #Pour ne pas revenir à la ligne
        self.image_list.setResizeMode(QListWidget.ResizeMode.Adjust) #Resize automatique
        self.image_list.setViewMode(QListWidget.ViewMode.IconMode) #Affichage par icone
        self.image_list.setIconSize(QSize(150, 150))
        self.image_list.setSpacing(10) # Espacement
        self.image_list.setStyleSheet("""
            QListWidget {
                background-color: #E6E6FA;
                border: 1px solid #00008B;
                border-radius: 10px;
            }
        """)
        for path in self.image_paths:
            item = QListWidgetItem()
            pixmap = QPixmap(path).scaledToWidth(150, Qt.TransformationMode.SmoothTransformation) # Ajuster la taille
            item.setIcon(QIcon(pixmap))
            item.setText(os.path.basename(path)) #Afficher le nom des images
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter) # Centrer le texte horizontalement
            self.image_list.addItem(item)
        main_layout.addWidget(self.image_list)

        # Barre de progression
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #00008B;
                border-radius: 5px;
                text-align: center;
                color: white;
                background-color: #4B0082;
            }

            QProgressBar::chunk {
                background-color: #800080;
                border-radius: 5px;
            }
        """)
        main_layout.addWidget(self.progress_bar)

        # Label "Traitement en cours..."
        self.processing_label = QLabel("Traitement en cours...", self)
        self.processing_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processing_label.setStyleSheet("color: #4B0082;")
        main_layout.addWidget(self.processing_label)

        # Layout horizontal pour les boutons
        buttons_layout = QHBoxLayout()
        buttons_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Bouton "Afficher le rapport" (initialement caché)
        self.show_report_button = QPushButton("Afficher le rapport", self)
        self.show_report_button.setStyleSheet("""
            QPushButton {
                background-color: #E6E6FA;
                border: 2px solid #00008B;
                color: #4B0082;
                padding: 8px 16px;
                text-align: center;
                text-decoration: none;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #D8BFD8;
            }
        """)
        self.show_report_button.clicked.connect(self.showReport)
        self.show_report_button.setIcon(QIcon(QDir.currentPath() + "/images/show_report.png"))  # Icône
        self.show_report_button.hide()
        buttons_layout.addWidget(self.show_report_button)

        # Bouton "Enregistrer le rapport" (initialement caché)
        self.save_report_button = QPushButton("Enregistrer le rapport", self)
        self.save_report_button.setStyleSheet("""
            QPushButton {
                background-color: #E6E6FA;
                border: 2px solid #00008B;
                color: #4B0082;
                padding: 8px 16px;
                text-align: center;
                text-decoration: none;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #D8BFD8;
            }
        """)
        self.save_report_button.clicked.connect(self.saveReport)
        self.save_report_button.setIcon(QIcon(QDir.currentPath() + "/images/save_report.png"))  # Icône
        self.save_report_button.hide()
        buttons_layout.addWidget(self.save_report_button)

        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)
        # Lancer le traitement au moment où la page est affichée

    def startProcessing(self):
        # Initialiser la barre de progression
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        # Lancer le processus dans un thread
        self.thread = ProcessingThread(self.image_paths)
        self.thread.progress_update.connect(self.updateProgress)
        self.thread.finished.connect(self.processingFinished)  # Connecté au signal finished

        self.thread.start()

    def updateProgress(self, value):
        self.progress_bar.setValue(value)

    def processingFinished(self, report_path):
        self.processing_label.setText("Traitement terminé!")
        self.show_report_button.show()
        self.save_report_button.show()
        self.report_path = report_path  # Sauvegarder le chemin du rapport

    def showReport(self):
        if hasattr(self, 'report_path') and self.report_path:
            if os.path.exists(self.report_path):
                os.startfile(self.report_path)  # Ouvrir le rapport avec l'application par défaut
            else:
                QMessageBox.warning(self, "Erreur", f"Le fichier '{self.report_path}' est introuvable.")
        else:
            QMessageBox.warning(self, "Erreur", "Rapport non disponible.")

    def saveReport(self):
        """Enregistrer le rapport au format PDF dans un dossier spécifique sans écraser."""
        if hasattr(self, 'report_path') and self.report_path:
            # Définir le dossier de destination
            reports_dir = os.path.join(os.getcwd(), "reports")  # Dossier "reports" dans le répertoire courant
            os.makedirs(reports_dir, exist_ok=True)  # Créer le dossier s'il n'existe pas

            # Générer un nom de fichier unique
            base_name = os.path.basename(self.report_path).rsplit('.', 1)[0]  # Nom sans extension
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_file_name = f"{base_name}_{timestamp}.pdf"
            destination_path = os.path.join(reports_dir, new_file_name)

            try:
                # Copy the PDF report to the destination
                shutil.copy2(self.report_path, destination_path)  # Use copy2 to preserve metadata
                QMessageBox.information(self, "Rapport", f"Rapport enregistré avec succès dans : {reports_dir}")
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Impossible d'enregistrer le rapport: {e}")

        else:
            QMessageBox.warning(self, "Erreur", "Aucun rapport disponible.")

    def showEvent(self, event):
        super().showEvent(event)
        # Lancer le traitement au moment où la page est affichée
        QTimer.singleShot(100, self.startProcessing)  # Attendre un peu avant de commencer

# Thread pour le traitement
class ProcessingThread(QThread):
    progress_update = pyqtSignal(int)
    finished = pyqtSignal(str)  # Signal pour emmetre le chemin du rapport

    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths

    def run(self):
        # Simuler le traitement avec une boucle
        report_path = "report.html"  # Chemin par defaut si la generation echoue
        try:
            for i in range(101):
                QThread.msleep(50)  # Pause pour simuler le traitement
                self.progress_update.emit(i)  # Emettre le signal de progression
            # Generer le rapport
            report_path = generate_report(self.image_paths)  # Creation du rapport
        except Exception as e:
            print(f"Erreur lors de la generation du rapport : {e}")
        finally:
            self.finished.emit(report_path)  # Emettre le signal avec le chemin du rapport




# ----- Page de Confirmation -----
import serial  # Importez la bibliothèque pySerial

# ----- Page de Confirmation -----


class ConfirmationPage(QWidget):
    page_shown = pyqtSignal()
    serial_connection_ready = pyqtSignal()

    def __init__(self, main_window, serial_port="COM4", baud_rate=9600):
        super().__init__()
        self.engine = pyttsx3.init()
        self.main_window = main_window
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.serial_connection = None
        self.current_language = "fr"  # Initialiser la langue par défaut
        self.main_window.home_page.language_combo.currentIndexChanged.connect(self.update_language)  # Connecter le signal
        # Initialisation des listes pour les GIFs
        base_path = "C:/Users/Danielle/Desktop/stage_N3/Vidéos"
        self.gif_paths = [
            os.path.abspath(f"{base_path}/drone2.gif"),
            os.path.abspath(f"{base_path}/drone2.gif"),
            os.path.abspath(f"{base_path}/drone3.gif"),
            os.path.abspath(f"{base_path}/drone3.gif"),
        ]

        for path in self.gif_paths:
            if not os.path.exists(path):
                print(f"Erreur: Le fichier n'existe pas à l'emplacement : {path}")

        # --- Initialisation des GIFs animés ---
        positions = [(0, 0), (0, 2), (2, 0), (2, 2)]  # Coins
        self.gif_labels = []
        self.movies = []

        # Initialisation des propriétés pour l'animation du texte
        self.typing_timer = QTimer(self)
        self.typing_text_fr = "Autoriser l'analyse du drone ?"  # Texte en français
        self.typing_text_en = "Allow drone analysis?" # Texte en anglais
        self.typing_text_es = "Permitir el análisis del dron?" # Texte en espagnol
        self.typed_text = ""
        self.typing_index = 0
        self.typing_timer.timeout.connect(self.typeNextChar)  # Connect here

        # List of Fonts and Colors
        self.fonts = [
            QFont("Arial", 24, QFont.Weight.Bold),
            QFont("Verdana", 24, QFont.Weight.Bold),
            QFont("Helvetica", 24, QFont.Weight.Bold),
            QFont("Times New Roman", 24, QFont.Weight.Bold),
        ]
        self.colors = [
            "#00008B",  # Dark Blue
            "#0000CD",  # Medium Blue
            "#191970",  # Midnight Blue
        ]

        self.initUI()
        self.serial_connection_ready.connect(self.enable_buttons)
        self.yes_button.setEnabled(False)
        self.no_button.setEnabled(False)

        self.main_window.home_page.language_combo.currentIndexChanged.connect(
            self.update_language)  # Connecter le signal

    def initUI(self):
        self.setWindowTitle("Analyse du drone")
        self.setGeometry(250, 150, 600, 300)
        self.setStyleSheet("QWidget { background-color: #FFFFFF; color: #000000; }")

        layout = QGridLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Titre - Initialement vide, il sera rempli par l'autotypage
        self.label = QLabel("")
        self.label.setFont(QFont("Arial", 36, QFont.Weight.Bold))  # Plus grand et plus gras
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet(
            """
            color: #00BFFF;
            text-shadow: 2px 2px 4px #808080;
            padding: 10px;
            border: 2px solid #00008B; /* Bordure bleu foncé */
        """
        )

        # Add drop shadow effect for relief
        shadow_effect = QGraphicsDropShadowEffect()
        shadow_effect.setBlurRadius(15)  # Adoucit les bords de l'ombre
        shadow_effect.setColor(QColor(0, 0, 0, 100))  # Couleur et transparence de l'ombre
        shadow_effect.setOffset(5, 5)  # Décalage de l'ombre (x, y)
        self.label.setGraphicsEffect(shadow_effect)

        # Create a sub-layout to contain the label
        label_layout = QVBoxLayout()
        label_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_layout.addWidget(self.label)
        layout.addLayout(label_layout, 1, 1)  # Centered in the grid

        # Positionnement des GIFs aux coins
        positions = [(0, 0), (0, 2), (2, 0), (2, 2)]  # Coins
        num_gifs = len(self.gif_paths)
        for i in range(num_gifs):  # Iterate only through available GIFs
            path = self.gif_paths[i]
            gif_label = QLabel(self)
            movie = QMovie(path)

            if not movie.isValid():
                print(f"Erreur: Impossible de charger le GIF à {path}")
                continue

            gif_label.setMovie(movie)
            gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            gif_label.setFixedSize(200, 150)  # Increased GIF box size
            movie.start()

            self.gif_labels.append(gif_label)
            self.movies.append(movie)

            row, col = (
                positions[i % len(positions)]
            )  # Wrap around positions if fewer positions than GIFs
            layout.addWidget(gif_label, row, col)

        # Boutons
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Bouton "Oui" (image + texte)
        self.yes_button_layout = QVBoxLayout()
        self.yes_button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.yes_button = QLabel(self)
        yes_pixmap = QPixmap(QDir.currentPath() + "/images/content.png").scaledToWidth(
            100, Qt.TransformationMode.SmoothTransformation
        )
        self.yes_button.setPixmap(yes_pixmap)
        self.yes_button.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.yes_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.yes_button.mousePressEvent = lambda event: self.onYesClicked(event)
        # add hover effects:
        self.yes_button.enterEvent = self.onYesButtonEnter
        self.yes_button.leaveEvent = self.onYesButtonLeave

        self.yes_label = QLabel("Oui", self)  # Stocker dans self pour la traduction
        self.yes_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.yes_label.setStyleSheet("color: #008000; font-weight: bold;")

        self.yes_button_layout.addWidget(self.yes_button)
        self.yes_button_layout.addWidget(self.yes_label)

        button_layout.addLayout(self.yes_button_layout)

        # Bouton "Non" (image + texte)
        self.no_button_layout = QVBoxLayout()
        self.no_button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.no_button = QLabel(self)
        no_pixmap = QPixmap(QDir.currentPath() + "/images/fache.png").scaledToWidth(
            100, Qt.TransformationMode.SmoothTransformation
        )  # Corrected
        self.no_button.setPixmap(no_pixmap)
        self.no_button.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.no_button.mousePressEvent = lambda event: self.onNoClicked(event)
        # add hover effects:
        self.no_button.enterEvent = self.onNoButtonEnter
        self.no_button.leaveEvent = self.onNoButtonLeave

        self.no_label = QLabel("Non", self)  # Stocker dans self pour la traduction
        self.no_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_label.setStyleSheet("color: #8B0000; font-weight: bold;")

        self.no_button_layout.addWidget(self.no_button)
        self.no_button_layout.addWidget(self.no_label)

        button_layout.addLayout(self.no_button_layout)

        # Layout des boutons en bas
        layout.addLayout(button_layout, 3, 1)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 2)
        layout.setRowStretch(2, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 2)
        layout.setColumnStretch(2, 1)

        self.setLayout(layout)
        self.translate_ui()  # Traduire l'interface au démarrage

        # Initialize animations for the buttons
        self.yes_animation = QPropertyAnimation(self.yes_button, b"geometry")
        self.yes_animation.setDuration(200)  # Animation duration in milliseconds

        self.no_animation = QPropertyAnimation(self.no_button, b"geometry")
        self.no_animation.setDuration(200)  # Animation duration in milliseconds

    def typeNextChar(self):
        typing_text = self.get_typing_text() # Recupere le text pour l'autotypage en fonction de la langue
        if self.typing_index < len(typing_text):
            self.typed_text += typing_text[self.typing_index]

            # Change the font and color randomly
            font = random.choice(self.fonts)
            color = random.choice(self.colors)

            self.label.setFont(font)
            self.label.setStyleSheet(
                f"""
                color: {color};
                text-shadow: 2px 2px 4px #808080;
                padding: 10px;
                border: 2px solid #00008B;
            """
            )

            self.label.setText(self.typed_text)
            self.typing_index += 1

        else:
            # Reset text and index after completion, stop timer, and restart
            self.typed_text = ""
            self.typing_index = 0
            self.label.setText("")  # Clear the label
            QTimer.singleShot(2000, self.start_typing_animation)

    def start_typing_animation(self):
        # start animation with the default font
        self.label.setFont(QFont("Arial", 36, QFont.Weight.Bold))
        self.label.setStyleSheet(
            """
            color: #00BFFF;
            text-shadow: 2px 2px 4px #808080;
            padding: 10px;
            border: 2px solid #00008B; /* Bordure bleu foncé */
        """
        )

        # Add drop shadow effect for relief
        shadow_effect = QGraphicsDropShadowEffect()
        shadow_effect.setBlurRadius(15)  # Adoucit les bords de l'ombre
        shadow_effect.setColor(QColor(0, 0, 0, 100))  # Couleur et transparence de l'ombre
        shadow_effect.setOffset(5, 5)  # Décalage de l'ombre (x, y)
        self.label.setGraphicsEffect(shadow_effect)

        self.typing_index = 0  # reset the index
        self.typed_text = ""
        self.typing_timer.start(150)

    def enable_buttons(self):
        self.yes_button.setEnabled(True)
        self.no_button.setEnabled(True)

    def showEvent(self, event):
        super().showEvent(event)
        self.page_shown.emit()
        print("showEvent called - ConfirmationPage")
        try:
            print("Attempting serial connection...")
            self.serial_connection = serial.Serial(self.serial_port, self.baud_rate)
            print(f"Connected to serial port {self.serial_port} at {self.baud_rate} baud")
            self.serial_connection_ready.emit()
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            self.serial_connection = None

        if self.serial_connection is None:
            print("Serial connection failed to initialize.")

        # Démarrer l'animation de l'autotypage au moment de l'affichage
        self.start_typing_animation()

    def closeEvent(self, event):
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.serial_connection.close()
                print("Serial port closed")
            except serial.SerialException as e:
                print(f"Error closing serial port: {e}")
        super().closeEvent(event)

    def onYesClicked(self, event=None):  # add event=None as argument
        print("Analyse du drone demandée.")
        threading.Thread(target=self.speak, args=(self.get_yes_response(),)).start()

        if self.main_window is not None:
            self.main_window.show_analyse_page()

            # Signal to AnalysePage to start the process
            self.main_window.analyse_page.start_analysis_sequence()

        else:
            print("main_window is none")

    def onNoClicked(self, event=None):  # add event=None as argument
        print("Analyse du drone annulée.")
        threading.Thread(target=self.speak, args=(self.get_no_response(),)).start()
        # Ici, tu peux ajouter le code pour annuler l'analyse du drone
        if self.serial_connection and self.serial_connection.is_open:
            print("About to send '0' to Arduino...")
            try:
                self.serial_connection.write(b"0")
                print("Successfully sent '0' to Arduino.")
            except serial.SerialException as e:
                print(f"Error sending data: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
        else:
            print("Serial connection not available or not open.")
        QApplication.instance().quit()

    def speak(self, text):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Erreur lors de la synthèse vocale : {e}")

    def onYesButtonEnter(self, event):
        self.animateButton(self.yes_button, 1.2)

    def onYesButtonLeave(self, event):
        self.animateButton(self.yes_button, 1.0)

    def onNoButtonEnter(self, event):
        self.animateButton(self.no_button, 1.2)

    def onNoButtonLeave(self, event):
        self.animateButton(self.no_button, 1.0)

    def animateButton(self, button, scale):
        original_rect = QRect(button.geometry())
        scaled_width = int(original_rect.width() * scale)
        scaled_height = int(original_rect.height() * scale)
        x = original_rect.x() - (scaled_width - original_rect.width()) // 2
        y = original_rect.y() - (scaled_height - original_rect.height()) // 2
        new_rect = QRect(x, y, scaled_width, scaled_height)

        if button is self.yes_button:
            self.yes_animation.setStartValue(button.geometry())
            self.yes_animation.setEndValue(new_rect)
            self.yes_animation.start()
        elif button is self.no_button:
            self.no_animation.setStartValue(button.geometry())
            self.no_animation.setEndValue(new_rect)
            self.no_animation.start()

    def translate_ui(self):
        if self.current_language == "fr":
            self.yes_label.setText("Oui")
            self.no_label.setText("Non")
        elif self.current_language == "en":
            self.yes_label.setText("Yes")
            self.no_label.setText("No")
        elif self.current_language == "es":
            self.yes_label.setText("Sí")
            self.no_label.setText("No")

    def get_typing_text(self):
        if self.current_language == "fr":
            return self.typing_text_fr
        elif self.current_language == "en":
            return self.typing_text_en
        elif self.current_language == "es":
            return self.typing_text_es
        return self.typing_text_fr  # Retourner le français par défaut si la langue n'est pas reconnue

    def get_yes_response(self):
         if self.current_language == "fr":
             return "D'accord, proccedons  à l'analyse."
         elif self.current_language == "en":
             return "Okay, let's proceed with the analysis."
         elif self.current_language == "es":
             return "De acuerdo, procedamos con el análisis."
         return "Okay, let's proceed with the analysis."  # Default to English if language not recognized

    def get_no_response(self):
        if self.current_language == "fr":
            return "Demande d'analyse annulée."
        elif self.current_language == "en":
            return "Analysis request canceled."
        elif self.current_language == "es":
            return "Solicitud de análisis cancelada."
        return "Analysis request canceled."  # Default to English if language not recognized

    def update_language(self, index):
        """
        Méthode appelée lorsqu'une nouvelle langue est sélectionnée dans le QComboBox de HomePage.
        Met à jour la langue courante et traduit l'interface utilisateur.
        """
        self.current_language = self.main_window.home_page.language_combo.itemData(index)
        self.translate_ui()
        # Redémarrer l'animation de l'autotypage avec le nouveau texte
        self.start_typing_animation()
class HomePage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.current_language = "fr"  # Langue par défaut: initialiser ici, AVANT init_tts
        self.init_tts()  # Initialiser la synthèse vocale
        self.initUI()

    def init_tts(self):
        """Initialise le moteur de synthèse vocale et configure les voix."""
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        self.french_voice_id = None
        self.english_voice_id = None
        self.spanish_voice_id = None  # Nouvelle variable pour l'espagnol

        for voice in self.voices:
            if "french" in voice.name.lower() or "français" in voice.name.lower():
                self.french_voice_id = voice.id
            if "english" in voice.name.lower():
                self.english_voice_id = voice.id
            if "spanish" in voice.name.lower() or "español" in voice.name.lower():  # Rechercher la voix espagnole
                self.spanish_voice_id = voice.id

        if self.french_voice_id is None:
            print("Voix française non trouvée.")
        if self.english_voice_id is None:
            print("Voix anglaise non trouvée.")
        if self.spanish_voice_id is None:  # Message pour l'espagnol
            print("Voix espagnole non trouvée.")

        self.set_voice()

    def set_voice(self):
        """Définit la voix de la synthèse vocale en fonction de la langue sélectionnée."""
        if self.current_language == "fr" and self.french_voice_id:
            self.engine.setProperty('voice', self.french_voice_id)
        elif self.current_language == "en" and self.english_voice_id:
            self.engine.setProperty('voice', self.english_voice_id)
        elif self.current_language == "es" and self.spanish_voice_id:  # Utiliser l'espagnol
            self.engine.setProperty('voice', self.spanish_voice_id)
        else:
            print("Langue non supportée ou voix non trouvée. Utilisation de la voix par défaut.")



    def initUI(self):
        # --- Layout principal ---
        main_layout = QGridLayout(self)
        main_layout.setContentsMargins(QMargins(20, 20, 20, 20))
        main_layout.setVerticalSpacing(10)

        # --- Chemins des GIFs et validation de leur existence ---
        base_path = "C:/Users/Danielle/Desktop/stage_N3/Vidéos"
        self.gif_paths = [
            os.path.abspath(f"{base_path}/drone1.gif"),
            os.path.abspath(f"{base_path}/drone2.gif"),
            os.path.abspath(f"{base_path}/drone3.gif"),
            os.path.abspath(f"{base_path}/drone3.gif")
        ]

        for path in self.gif_paths:
            if not os.path.exists(path):
                print(f"Erreur: Le fichier n'existe pas à l'emplacement : {path}")

        # --- Initialisation des GIFs animés ---
        positions = [(0, 0), (0, 2), (2, 0), (2, 2)]  # Coins
        self.gif_labels = []
        self.movies = []
        for i, path in enumerate(self.gif_paths):
            gif_label = QLabel(self)
            movie = QMovie(path)

            if not movie.isValid():
                print(f"Erreur: Impossible de charger le GIF à {path}")
                continue

            gif_label.setMovie(movie)
            gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            gif_label.setFixedSize(200, 150)
            movie.start()

            self.gif_labels.append(gif_label)
            self.movies.append(movie)

            row, col = positions[i]
            main_layout.addWidget(gif_label, row, col)

        # --- Style transparent pour les widgets ---
        transparent_style = """
            background-color: rgba(0, 0, 0, 0);
        """

        # --- Bloc de Texte ---
        text_layout = QVBoxLayout()  # Layout vertical pour le bloc de texte
        text_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- Titre ---
        self.title = QLabel("Drone AI Vision", self)
        self.title.setFont(QFont("Helvetica", 36, QFont.Weight.Bold))
        title_layout = QHBoxLayout()
        title_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(self.title)
        self.title.setStyleSheet(f"""
            color: #7030A0; /* Violet Foncé */
            {transparent_style}
            text-shadow: 1px 1px 2px #301050,
                         2px 2px 4px #502070,
                         3px 3px 6px #7030A0;
        """)
        text_layout.addLayout(title_layout)

        # --- Sous-titre ---
        self.subtitle = QLabel("Détection d'anomalies visuelles par IA", self)
        self.subtitle.setFont(QFont("Helvetica", 16))
        subtitle_layout = QHBoxLayout()
        subtitle_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_layout.addWidget(self.subtitle)
        self.subtitle.setStyleSheet(f"""
            color: #7030A0; /* Violet Brillant */
            {transparent_style}
            text-shadow: 1px 1px 2px #301050,
                         2px 2px 4px #502070;
        """)
        text_layout.addLayout(subtitle_layout)

        # --- Description ---
        self.description = QLabel("Analyse intelligente des images de drones pour identifier rapidement les anomalies et les problèmes potentiels.", self)
        self.description.setFont(QFont("Helvetica", 12))
        description_layout = QHBoxLayout()
        description_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description_layout.addWidget(self.description)
        self.description.setStyleSheet(f"""
            color: #003366; /* Bleu Foncé */
            padding: 20px;
            {transparent_style}
            text-shadow: 1px 1px 2px #001020,
                         2px 2px 4px #002040;
        """)
        text_layout.addLayout(description_layout)

        # --- Image de drone et Bouton ---
        bottom_layout = QVBoxLayout()
        bottom_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- Image de drone ---
        self.drone_image = QLabel(self)
        image_path_drone = QDir.currentPath() + "/images/drone.png"  # Remplacez par votre image
        pixmap_drone = QPixmap(image_path_drone)
        if pixmap_drone.isNull():
            print(f"Erreur: Impossible de charger l'image à {image_path_drone}")
        else:
            self.drone_image.setPixmap(pixmap_drone.scaledToWidth(150, Qt.TransformationMode.SmoothTransformation))
        self.drone_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drone_image.setStyleSheet(transparent_style)
        bottom_layout.addWidget(self.drone_image)

        # --- ComboBox pour le choix de la langue ---
        self.language_combo = QComboBox(self)
        self.language_combo.addItem("Français", "fr")
        self.language_combo.addItem("English", "en")
        self.language_combo.addItem("Español", "es")  # Remplacer mandarin par espagnol
        self.language_combo.currentIndexChanged.connect(self.change_language)
        bottom_layout.addWidget(self.language_combo)



        # --- Bouton "Démarrer" ---
        self.start_button = QPushButton("Démarrer", self)
        self.start_button.setFont(QFont("Helvetica", 14, QFont.Weight.Bold))
        self.start_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)  # Etirement horizontal
        self.start_button.setStyleSheet(f"""
            QPushButton {{
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                 stop:0 #003366, stop:1 #001A33); /* Gradient bleu foncé */
                border: 2px solid #00FFFF;
                color: #FFFFFF;
                padding: 10px 20px;
                border-radius: 5px;
                text-shadow: 1px 1px 2px #001020,
                             2px 2px 4px #002040; /* Multiples ombres portées */
            }}
            QPushButton:hover {{
                color: #00FFFF; /* Texte Cyan au survol */
                border: 2px solid #7030A0; /* Bordure Violet au survol */
            }}
        """)
        bottom_layout.addWidget(self.start_button)

        # --- Ajouter les widgets au layout principal ---
        # Les GIFs occupent les coins
        positions = [(0, 0), (0, 2), (2, 0), (2, 2)]
        for i, path in enumerate(self.gif_paths):
            gif_label = QLabel(self)
            movie = QMovie(path)

            if not movie.isValid():
                print(f"Erreur: Impossible de charger le GIF à {path}")
                continue

            gif_label.setMovie(movie)
            gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            gif_label.setFixedSize(200, 150)
            movie.start()

            row, col = positions[i]
            main_layout.addWidget(gif_label, row, col)

        main_layout.addLayout(text_layout, 1, 1)
        main_layout.addLayout(bottom_layout, 3, 1)
        main_layout.setRowStretch(0, 1)
        main_layout.setRowStretch(1, 2)
        main_layout.setRowStretch(2, 1)
        main_layout.setColumnStretch(0, 1)
        main_layout.setColumnStretch(1, 2)
        main_layout.setColumnStretch(2, 1)

        # --- Connecter le bouton à la méthode showConfirmationPage ---
        self.start_button.clicked.connect(self.showConfirmationPage)

        self.setLayout(main_layout)
        self.translate_ui() # Initialiser l'interface dans la langue par défaut



    def change_language(self, index):
        """
        Méthode appelée lorsqu'une nouvelle langue est sélectionnée dans le QComboBox.
        Met à jour la langue courante et traduit l'interface utilisateur.
        """
        self.current_language = self.language_combo.itemData(index) # Récupérer les données (code langue)
        self.set_voice()  # Mettre à jour la voix
        self.translate_ui()
        self.speak_welcome_message()

    def translate_ui(self):
        """Traduit les éléments de l'interface utilisateur en fonction de la langue sélectionnée."""
        if self.current_language == "fr":
            self.title.setText("Drone AI Vision")
            self.subtitle.setText("Détection d'anomalies visuelles par IA")
            self.description.setText(
                "Analyse intelligente des images de drones pour identifier rapidement les anomalies et les problèmes potentiels.")
            self.start_button.setText("Démarrer")
        elif self.current_language == "en":
            self.title.setText("Drone AI Vision")
            self.subtitle.setText("Visual Anomaly Detection by AI")
            self.description.setText(
                "Intelligent analysis of drone images to quickly identify anomalies and potential problems.")
            self.start_button.setText("Start")
        elif self.current_language == "es":  # Traduction en espagnol
            self.title.setText("Drone AI Vision")  # Ajuster si une traduction appropriée existe
            self.subtitle.setText("Detección de anomalías visuales por IA")
            self.description.setText(
                "Análisis inteligente de imágenes de drones para identificar rápidamente anomalías y problemas potenciales.")
            self.start_button.setText("Comenzar")
        # Ajouter d'autres langues si nécessaire

    def get_welcome_message(self):
        """Retourne le message de bienvenue dans la langue appropriée."""
        if self.current_language == "fr":
            return "Bienvenue dans Drone AI Vision, votre IA de détection des anomalies visuelles des drones. Pour commencer, appuyez sur le bouton Démarrer."
        elif self.current_language == "en":
            return "Welcome to Drone AI Vision, your AI for detecting visual anomalies in drones. To start, press the Start button."
        elif self.current_language == "es":  # Message de bienvenue en espagnol
            return "Bienvenido a Drone AI Vision, su IA para la detección de anomalías visuales en drones. Para comenzar, pulse el botón Comenzar."
        else:
            return "Welcome to Drone AI Vision."  # Message par défaut


    def speak_welcome_message(self):
        """Prononce le message de bienvenue dans la langue sélectionnée."""
        welcome_message = self.get_welcome_message()
        self.speak(welcome_message)



    def showConfirmationPage(self):
        print("showConfirmationPage from HomePage called")# Pour deboguer
        self.main_window.show_confirmation_page()
        print("self.main_window.show_confirmation_page() executed (Depuis HomePage)")

    def speak(self, text):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Erreur lors de la synthèse vocale : {e}")


# ----- Programme principal -----
# ----- Programme principal -----
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone AI")
        self.setGeometry(200, 100, 900, 550)
        self.stacked_widget = QStackedWidget(self)
        self.engine = pyttsx3.init()
        self.serial_port = "COM4"  # Remplacez par le bon port série
        self.baud_rate = 9600        # Vitesse de communication série

        self.home_page = HomePage(self)  # Passez une référence à 'self'

        self.confirmation_page = ConfirmationPage(self, self.serial_port, self.baud_rate)  # Passez une référence à 'self' et les informations du port série
        self.analyse_page = AnalysePage(self) #Creer la page d'analyse
        self.processing_page = None # Initialize processing_page to None

        self.stacked_widget.addWidget(self.home_page)
        self.stacked_widget.addWidget(self.confirmation_page)
        self.stacked_widget.addWidget(self.analyse_page)

        layout = QVBoxLayout(self)
        layout.addWidget(self.stacked_widget)
        self.setLayout(layout)
        self.setStyleSheet("background-color: #FFFFFF; color: #000000;") #Fond blanc self.setStyleSheet("background-color: #FFFFFF; color: #000000;") #Fond blanc

        #Connecter le signal de la page de confirmation a la methode speak.
        self.confirmation_page.page_shown.connect(self.on_confirmation_page_shown)


    def show_home_page(self):
        self.stacked_widget.setCurrentWidget(self.home_page)

    def show_confirmation_page(self):
        self.stacked_widget.setCurrentWidget(self.confirmation_page)

    def show_analyse_page(self):
        self.stacked_widget.setCurrentWidget(self.analyse_page)

    def show_processing_page(self, image_paths): #method gets path and creates to the stack

        if self.processing_page is not None:
             self.processing_page.deleteLater()#delete if exists
        self.processing_page = ProcessingPage(image_paths)
        self.stacked_widget.addWidget(self.processing_page) #adds if doesnt exists
        self.stacked_widget.setCurrentWidget(self.processing_page)

    def on_confirmation_page_shown(self):
        #Lancer la synthèse vocale sur le thread principale apres l'affichage de la page
        QTimer.singleShot(500, lambda: self.confirmation_page.speak("Souhaitez-vous faire analyser votre drone ? Oui ou non"))





if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 1. Afficher l'écran de chargement
    splash = SplashScreen()
    splash.show()

    # 2. Créer et initialiser la fenêtre principale dans un thread séparé
    def initialize_main_window():
        global main_window
        main_window = MainWindow()  # Instance de MainWindow ici
        main_window.show()
        splash.close()

    # 3. Utiliser un QTimer pour retarder l'initialisation et l'affichage de la fenêtre principale
    QTimer.singleShot(3000, initialize_main_window)

    sys.exit(app.exec())