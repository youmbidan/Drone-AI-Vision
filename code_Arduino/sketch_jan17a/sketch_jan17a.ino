#include <Stepper.h>

// Nombre de pas par tour pour le moteur 28BYJ-48
const int stepsPerRevolution = 2048;

// Initialisation du moteur (broches IN1 à IN4 du driver ULN2003)
Stepper myStepper(stepsPerRevolution, 8, 10, 9, 11);

void setup() {
  Serial.begin(9600);
  myStepper.setSpeed(10);  // Vitesse en RPM (tr/min)
}

void loop() {
  // Vérifie s'il y a des données disponibles sur le port série
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n'); // Lire jusqu'à la fin de la ligne
    command.trim(); // Supprimer les espaces inutiles

    if (command.length() > 0) {
      if (command == "0") {
        // Arrêter le moteur (si nécessaire)
        Serial.println("Moteur arrêté");
      } else {
        // Essayer de convertir la commande en un angle
        int angle = command.toInt();

        if (angle != 0) {
          // Calculer le nombre de pas nécessaires pour l'angle
          float stepsFloat = ((float)angle / 360.0) * stepsPerRevolution; // Calculer le nombre de pas comme un float
          int steps = round(stepsFloat);  // Arrondir au plus proche entier

          Serial.print("Tourner de ");
          Serial.print(angle);
          Serial.print(" degrés (");
          Serial.print(steps);
          Serial.println(" pas)");

          // Faire tourner le moteur
          myStepper.step(steps);

          delay(500); // Petite pause pour assurer l'arrêt

          Serial.println("R"); // Envoyer la confirmation
        } else {
          Serial.println("Commande invalide");
        }
      }
    }
  }
}
