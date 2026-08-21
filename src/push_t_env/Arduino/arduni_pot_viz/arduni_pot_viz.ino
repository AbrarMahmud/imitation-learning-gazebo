// ---------------- Pin definitions ---------------
// ---------------- Pin definitions ----------------
const int pot1Pin = A3;
const int pot2Pin = A4;
const int pot3Pin = A5;

// ---------------- Calibration (raw ADC -> radians) ----------------
// Edit these per-pot values to match your measured min/max raw readings
// and desired output angle range.
struct PotCalib {
  float rawMin;
  float rawMax;
  float outMin;
  float outMax;
};

PotCalib potCalib[3] = {
  { 820.0,  170.0, -3.1416/2, 3.1416/2 },  // pot1: raw DOWN  -> angle UP
  { 170.0,  820.0, -3.1416/2, 3.1416/2 },  // pot2: raw UP    -> angle UP
  { 170.0,  820.0, -3.1416/2, 3.1416/2 }   // pot3: raw UP    -> angle UP
};
// Generic float mapping (like Arduino's map(), but works with floats)
// Works even if inMin > inMax (i.e. a "reversed" pot calibration).
float mapFloat(float x, float inMin, float inMax, float outMin, float outMax) {
  // clamp input to the calibration range so output never leaves [outMin, outMax],
  // regardless of whether inMin/inMax are given in ascending or descending order
  float lo = min(inMin, inMax);
  float hi = max(inMin, inMax);
  if (x < lo) x = lo;
  if (x > hi) x = hi;
  return (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
}

// ---------------- Low-pass (EMA) filter ----------------
// Smaller alpha = smoother but slower response, larger alpha = snappier but noisier.
// Tune 0.0 - 1.0 to taste.
const float FILTER_ALPHA = 0.15;
float filteredRaw[3] = { 0, 0, 0 };
bool filterInit = false;

float lowPassFilter(float newValue, float &prevFiltered) {
  prevFiltered = FILTER_ALPHA * newValue + (1.0 - FILTER_ALPHA) * prevFiltered;
  return prevFiltered;
}

float positions[3];
uint32_t seqCounter = 0;

void setup() {
  Serial.begin(9600);
}

void loop() {
  // 1. Read raw ADC values
  int raw1 = analogRead(pot1Pin);
  int raw2 = analogRead(pot2Pin);
  int raw3 = analogRead(pot3Pin);

  // Initialize filter state on first pass to avoid a startup ramp-in
  if (!filterInit) {
    filteredRaw[0] = raw1;
    filteredRaw[1] = raw2;
    filteredRaw[2] = raw3;
    filterInit = true;
  }

  // 2. Smooth the raw readings (prevents sudden jumps / noise)
  float f1 = lowPassFilter((float)raw1, filteredRaw[0]);
  float f2 = lowPassFilter((float)raw2, filteredRaw[1]);
  float f3 = lowPassFilter((float)raw3, filteredRaw[2]);

  // 3. Map filtered raw values to each joint's radian range
  positions[0] = mapFloat(f1, potCalib[0].rawMin, potCalib[0].rawMax,
                           potCalib[0].outMin, potCalib[0].outMax);
  positions[1] = mapFloat(f2, potCalib[1].rawMin, potCalib[1].rawMax,
                           potCalib[1].outMin, potCalib[1].outMax);
  positions[2] = mapFloat(f3, potCalib[2].rawMin, potCalib[2].rawMax,
                           potCalib[2].outMin, potCalib[2].outMax);

  // 4. Fill header and publish
  Serial.print("Pot1:");
  Serial.print(positions[0]);
  Serial.print(",");

  Serial.print("Pot2:");
  Serial.print(positions[1]);
  Serial.print(",");

  Serial.print("Pot3:");
  Serial.println(positions[2]);
  delay(50); // ~20 Hz publish rate
}