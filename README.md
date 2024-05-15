# Task 1

Beim Start der gather-data.py kann man zunächst die Sportart auswählen. Daraufhin werden für die Datensammlung 10000 Datenreihen erhoben, sobald man auf den Button1 am M5 drückt. Eine Erhebung dauert ca. 10 Sekunden. Namensvergabe der .csv und Speicherung passiert danach automatisch.

Für das Resamplen habe ich das Skript etwas angepasst, dass man es nur ein einziges mal durchlaufen lassen muss.

# Task 2

Im Activity-Recognizer wird eine Klasse erstellt, die die Daten einliest, umwandelt und einen Klassifizierer trainiert. Ich habe mich dafür entschieden, die Aktivitäten auf Basis der Frequenz der einzelnen Messwerte per FFT zu bestimmen. Im Fitness-Trainer wird immer ein ca. 5-sekündiger Datensatz aufgenommen und daraus dann die Aktivität bestimmt. Diese wird dann über pyglet graphisch dargestellt.