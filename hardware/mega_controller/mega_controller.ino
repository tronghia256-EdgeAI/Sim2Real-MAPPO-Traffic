/*
 * mega_controller.ino
 * ===================
 * Adaptive RYG traffic light controller for Arduino Mega.
 *
 * Serial protocol (from host Python):
 *   <S0,S1,S2,S3,S4,S5,S6,S7>\n
 *   S_i in {0, 2}  =>  0=RED  2=GREEN   (Python sends only RED/GREEN)
 *   Arduino handles yellow transition internally.
 *   8 directions: [tls0_dir0, tls0_dir1, tls0_dir2, tls0_dir3,
 *                  tls1_dir0, tls1_dir1, tls1_dir2, tls1_dir3]
 *
 * Transition logic (per direction):
 *   GREEN → RED  :  GREEN → YELLOW (3 s) → all-RED gap (1 s) → RED
 *   RED   → RED  :  stays RED during the yellow/gap interval
 *   RED   → GREEN:  waits for yellow+gap interval, then goes GREEN
 *
 * Pin layout (Arduino Mega digital pins 22-45):
 *   Dir 0: R=22  Y=23  G=24
 *   Dir 1: R=25  Y=26  G=27
 *   Dir 2: R=28  Y=29  G=30
 *   Dir 3: R=31  Y=32  G=33
 *   Dir 4: R=34  Y=35  G=36
 *   Dir 5: R=37  Y=38  G=39
 *   Dir 6: R=40  Y=41  G=42
 *   Dir 7: R=43  Y=44  G=45
 */

#define NUM_DIRS      8
#define YELLOW_MS     3000   // yellow duration (ms)
#define RED_GAP_MS    0   // all-red gap after yellow (ms)

const byte R_PINS[NUM_DIRS] = {22, 25, 28, 31, 34, 37, 40, 43};
const byte Y_PINS[NUM_DIRS] = {23, 26, 29, 32, 35, 38, 41, 44};
const byte G_PINS[NUM_DIRS] = {24, 27, 30, 33, 36, 39, 42, 45};

const byte STATE_RED    = 0;
const byte STATE_YELLOW = 1;
const byte STATE_GREEN  = 2;

const byte numChars = 64;
char receivedChars[numChars];
char tempChars[numChars];

// Current live states shown on hardware
byte lightStates[NUM_DIRS] = {STATE_RED, STATE_RED, STATE_RED, STATE_RED,
                               STATE_RED, STATE_RED, STATE_RED, STATE_RED};

boolean newData = false;

void setup() {
    Serial.begin(115200);
    for (int i = 0; i < NUM_DIRS; i++) {
        pinMode(R_PINS[i], OUTPUT);
        pinMode(Y_PINS[i], OUTPUT);
        pinMode(G_PINS[i], OUTPUT);
    }
    applyAllRed();
    Serial.println("Arduino Mega ready — RYG adaptive phase control.");
}

void loop() {
    recvWithStartEndMarkers();
    if (newData == true) {
        strcpy(tempChars, receivedChars);
        byte targetStates[NUM_DIRS];
        parseData(targetStates);
        applyWithTransition(targetStates);
        newData = false;
    }
}

// ── Serial receive (non-blocking) ───────────────────────────────────────────

void recvWithStartEndMarkers() {
    static boolean recvInProgress = false;
    static byte ndx = 0;
    char startMarker = '<';
    char endMarker   = '>';
    char rc;

    while (Serial.available() > 0 && newData == false) {
        rc = Serial.read();
        if (recvInProgress == true) {
            if (rc != endMarker) {
                receivedChars[ndx] = rc;
                ndx++;
                if (ndx >= numChars) ndx = numChars - 1;
            } else {
                receivedChars[ndx] = '\0';
                recvInProgress = false;
                ndx = 0;
                newData = true;
            }
        } else if (rc == startMarker) {
            recvInProgress = true;
        }
    }
}

// ── Parse "<S0,S1,...,S7>" into targetStates[] ───────────────────────────────

void parseData(byte targetStates[NUM_DIRS]) {
    char *tok = strtok(tempChars, ",");
    for (int i = 0; i < NUM_DIRS; i++) {
        if (tok != NULL) {
            int val = atoi(tok);
            // Accept only RED(0) or GREEN(2); anything else → RED
            targetStates[i] = (val == STATE_GREEN) ? STATE_GREEN : STATE_RED;
            tok = strtok(NULL, ",");
        } else {
            targetStates[i] = STATE_RED;
        }
    }
}

// ── Transition logic ─────────────────────────────────────────────────────────
//
//  Step 1 — Yellow interval (YELLOW_MS):
//    GREEN → RED  : show YELLOW
//    RED   → *    : stay RED
//
//  Step 2 — All-red gap (RED_GAP_MS)
//
//  Step 3 — Apply target states

void applyWithTransition(byte targetStates[NUM_DIRS]) {
    // Check if any direction is switching GREEN → RED
    boolean needsTransition = false;
    for (int i = 0; i < NUM_DIRS; i++) {
        if (lightStates[i] == STATE_GREEN && targetStates[i] == STATE_RED) {
            needsTransition = true;
            break;
        }
    }

    if (needsTransition) {
        // Step 1: yellow only for GREEN→RED, keep RED for everything else
        for (int i = 0; i < NUM_DIRS; i++) {
            if (lightStates[i] == STATE_GREEN && targetStates[i] == STATE_RED) {
                lightStates[i] = STATE_YELLOW;
            }
            // Directions that are RED stay RED (no change needed)
        }
        executeHardware();
        delay(YELLOW_MS);

        // Step 2: all-red gap
        for (int i = 0; i < NUM_DIRS; i++) {
            lightStates[i] = STATE_RED;
        }
        executeHardware();
        delay(RED_GAP_MS);

        // Flush commands that arrived during the blocking transition
        while (Serial.available()) Serial.read();
    }

    // Step 3: apply target states
    for (int i = 0; i < NUM_DIRS; i++) {
        lightStates[i] = targetStates[i];
    }
    executeHardware();
}

// ── Drive LED outputs based on lightStates[] ────────────────────────────────

void executeHardware() {
    for (int i = 0; i < NUM_DIRS; i++) {
        digitalWrite(R_PINS[i], lightStates[i] == STATE_RED    ? HIGH : LOW);
        digitalWrite(Y_PINS[i], lightStates[i] == STATE_YELLOW ? HIGH : LOW);
        digitalWrite(G_PINS[i], lightStates[i] == STATE_GREEN  ? HIGH : LOW);
    }
}

// ── Safety helper ────────────────────────────────────────────────────────────

void applyAllRed() {
    for (int i = 0; i < NUM_DIRS; i++) {
        lightStates[i] = STATE_RED;
    }
    executeHardware();
}
