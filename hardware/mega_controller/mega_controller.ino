/*
 * mega_controller.ino
 * ===================
 * Adaptive RYG traffic light controller for Arduino Mega.
 *
 * Serial protocol (from host Python):
 *   <S0,S1,S2,S3,S4,S5,S6,S7>\n
 *   S_i in {0, 1, 2}  =>  0=RED  1=YELLOW  2=GREEN
 *   8 directions: [tls0_dir0, tls0_dir1, tls0_dir2, tls0_dir3,
 *                  tls1_dir0, tls1_dir1, tls1_dir2, tls1_dir3]
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

#define NUM_DIRS 8

const byte R_PINS[NUM_DIRS] = {22, 25, 28, 31, 34, 37, 40, 43};
const byte Y_PINS[NUM_DIRS] = {23, 26, 29, 32, 35, 38, 41, 44};
const byte G_PINS[NUM_DIRS] = {24, 27, 30, 33, 36, 39, 42, 45};

const byte STATE_RED    = 0;
const byte STATE_YELLOW = 1;
const byte STATE_GREEN  = 2;

const byte numChars = 64;
char receivedChars[numChars];
char tempChars[numChars];

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

    // Boot state: all red
    applyAllRed();
    Serial.println("Arduino Mega ready — RYG adaptive phase control.");
}

void loop() {
    recvWithStartEndMarkers();
    if (newData == true) {
        strcpy(tempChars, receivedChars);
        parseData();
        executeHardware();
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

// ── Parse "<S0,S1,...,S7>" into lightStates[] ────────────────────────────────

void parseData() {
    char *tok = strtok(tempChars, ",");
    for (int i = 0; i < NUM_DIRS; i++) {
        if (tok != NULL) {
            int val = atoi(tok);
            // Clamp to valid range
            if (val < 0 || val > 2) val = STATE_RED;
            lightStates[i] = (byte)val;
            tok = strtok(NULL, ",");
        }
    }
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
