/**
 * @file mask_serial.ino
 * @brief Freddie 2.0 mask firmware (ESP32-S3 + PCA9685) — FreeRTOS tasked
 *
 * ── What this does ─────────────────────────────────────────────────────────────
 * • Drives 16 servo channels @50 Hz via PCA9685 with a 20 µs/frame slew limiter.
 * • Restores last pose/base from NVS; clamps to per-channel limits.
 * • One pushbutton = instant NEUTRAL (failsafe), no serial spam.
 * • PCA9685 OE pin handled for quiet startup (optional).
 *
 * ── Channels (index) ─────────────────────────────────────────────────────────
 * 0 - 12 facial, 13-15 neck
 * Names for reference:
 * ["L brow V","L brow A","R brow V","R brow A","R eye X","R eye Y",
 *  "R eyelid","L eye X","L eye Y","L eyelid","Jaw","R mouth",
 *  "L mouth","Neck yaw","Neck R","Neck L"]
 *
 * ── Serial protocol (newline-terminated) ──────────────────────────────────────
 *   PING                         -> PONG
 *   NEUTRAL                      -> go to base neutral pose
 *   FACE neutral                 -> same as NEUTRAL (stub)
 *   POSE  <id> <us>              -> set channel (slewed)
 *   RAW   <id> <us>              -> immediate write (no slew) + set targets
 *   POSES us0,us1,...,(us13|us15)-> set 13 legacy or 16 full targets (slewed)
 *   SAVE                         -> persist current pose + base to NVS
 *   SETBASE us0,...,(us13|us15)  -> set base (Poker) and save (13 or 16 values)
 *   PRINTPOS                     -> print key and all channels
 *   CLEARNVS                     -> clear NVS keys (base/pose)
 *   LOADPOS                      -> reload from NVS and reapply to hardware
 *   FACE <other>                 -> ERR face not found

 * @author  Rambod Taherian
 * @version 1.0.0
 * @date    2025-09-17
 * @source  https://github.com/rambodt/Freddie-2.0
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <Preferences.h>

// FreeRTOS (explicit includes for clarity on ESP32 Arduino)
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

// ── PCA9685 / timing ─────────────────────────────────────────────────────────
#define N_SERVO        16
#define SERVO_FREQ     50
#define MIN_US_GLOBAL  500
#define MAX_US_GLOBAL  2400
#define MAX_STEP_US    20                // slew step per 20 ms frame

#define SDA_PIN        5
#define SCL_PIN        4
#define USE_OE         1
#if USE_OE
  #define OE_PIN       6                 // active-LOW
#endif

#define BUTTON_PIN     16                // INPUT_PULLUP, press=LOW

// ── Channel map (min, max, defaultBase, name) ──────────────────────────
// {   930,  1050, 1020, "L_BROW_V" },  // Ch 0:  Left eyebrow vertical
// {  1041,  1800, 1420, "L_BROW_A" },  // Ch 1:  Left eyebrow angle
// {  2070,  2200, 2100, "R_BROW_V" },  // Ch 2:  Right eyebrow vertical
// {  1250,  1900, 1510, "R_BROW_A" },  // Ch 3:  Right eyebrow angle
// {  1039,  1600, 1290, "R_EYE_X"  },  // Ch 4:  Right eye X-axis
// {   890,  1510, 1210, "R_EYE_Y"  },  // Ch 5:  Right eye Y-axis
// {  1200,  2100, 1560, "R_EYELID" },  // Ch 6:  Right eyelid
// {  1557,  2180, 1880, "L_EYE_X"  },  // Ch 7:  Left eye X-axis
// {  1280,  1871, 1620, "L_EYE_Y"  },  // Ch 8:  Left eye Y-axis
// {  1139,  2000, 1670, "L_EYELID" },  // Ch 9:  Left eyelid
// {  1216,  1750, 1260, "JAW"      },  // Ch 10: Jaw
// {   660,  2200, 1310, "R_MOUTH"  },  // Ch 11: Right mouth corner
// {   500,  1820, 1225, "L_MOUTH"  },  // Ch 12: Left mouth corner
// {   500,  2300, 1400, "NECK_YAW" },  // Ch 13: Neck yaw (left-right)
// {   500,  2400, 1450, "NECK_R"   },  // Ch 14: Neck right motor
// {   500,  2400, 1450, "NECK_L"   },  // Ch 15: Neck left motor

// ── Per-channel limits (FINAL) ───────────────────────────────────────────────
struct Limits { uint16_t minUs, maxUs; };
static const Limits lim[N_SERVO] = {
  { 930,1050}, {1041,1800}, {2070,2200}, {1250,1900},
  {1039,1600}, { 890,1510}, {1200,2100}, {1557,2180},
  {1280,1871}, {1139,2000}, {1216,1750}, { 660,2200},
  { 500,1820}, { 500,2300}, { 500,2400}, { 500,2400}
};

// ── Default Poker / Neutral pose (matches PC script) ─────────────────────────
static const uint16_t DEFAULT_BASE[N_SERVO] = {
  1020,1420,2100,1510,1290,1210,1560,1880,
  1620,1670,1260,1310,1225,1400,1450,1450
};

// ── Globals ──────────────────────────────────────────────────────────────────
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);
Preferences prefs;

// Base pose used by NEUTRAL and as a starting point for faces; persisted in NVS
static uint16_t baseUs[N_SERVO];

// Target and current (slew) positions.
// Accessed from multiple tasks; protect with a spinlock mux (very short CS).
static volatile uint16_t targetUs[N_SERVO];
static volatile uint16_t currUs[N_SERVO];
static portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;

// ── FreeRTOS objects ─────────────────────────────────────────────────────────
/** Queue of ASCII command lines from SerialRx → Cmd parser. */
typedef struct { char line[256]; } LineMsg;
static QueueHandle_t qLines = nullptr;

// ── Forward declarations ─────────────────────────────────────────────────────
static void TaskSlew(void* arg);
static void TaskSerialRx(void* arg);
static void TaskCmd(void* arg);
static void TaskButton(void* arg);

static void handleLine(String line);
static void neutralTargets();
static void writeAllCurr();
static void setOE(bool enabled);

// ── Utilities ────────────────────────────────────────────────────────────────
static inline uint16_t clampu(uint16_t x, uint16_t lo, uint16_t hi){
  if(x < lo) return lo; if(x > hi) return hi; return x;
}
static inline uint16_t clampServo(uint8_t ch, uint16_t us){
  uint16_t v = clampu(us, MIN_US_GLOBAL, MAX_US_GLOBAL);
  v = clampu(v, lim[ch].minUs, lim[ch].maxUs);
  return v;
}

static void writeServoUs(uint8_t ch, uint16_t us){
  pwm.writeMicroseconds(ch, us);
}

static void printKeyChannels(const char* tag){
  Serial.printf("[%s] Rmouth(ch11)=%u  Lmouth(ch12)=%u  Jaw(ch10)=%u\n",
                tag, (unsigned)currUs[11], (unsigned)currUs[12], (unsigned)currUs[10]);
}
static void printAllChannels(const char* tag){
  Serial.print("["); Serial.print(tag); Serial.print("] ");
  for(uint8_t i=0;i<N_SERVO;i++){
    Serial.print(i); Serial.print("="); Serial.print((unsigned)currUs[i]);
    if(i != N_SERVO-1) Serial.print(", ");
  }
  Serial.println();
}

static void savePose(){
  prefs.begin("servos", false);
  prefs.putBytes("pos",  (const void*)currUs, sizeof(currUs));
  prefs.putBytes("base", (const void*)baseUs, sizeof(baseUs));
  prefs.end();
}

static bool loadPose(){
  prefs.begin("servos", true);
  bool okPos  = (prefs.getBytesLength("pos")  == sizeof(currUs));
  bool okBase = (prefs.getBytesLength("base") == sizeof(baseUs));

  // Load base or fall back to DEFAULT_BASE (Poker)
  if(okBase){
    uint16_t tmpB[N_SERVO];
    prefs.getBytes("base", (void*)tmpB, sizeof(tmpB));
    for(uint8_t i=0;i<N_SERVO;i++) baseUs[i] = clampServo(i, tmpB[i]);
  } else {
    for(uint8_t i=0;i<N_SERVO;i++) baseUs[i] = clampServo(i, DEFAULT_BASE[i]);
  }

  // If a last pose was saved, load it (clamped). Otherwise use base as curr.
  if(okPos){
    uint16_t tmp[N_SERVO];
    prefs.getBytes("pos", (void*)tmp, sizeof(tmp));
    for(uint8_t i=0;i<N_SERVO;i++){
      uint16_t v = clampServo(i, tmp[i]);
      currUs[i] = v; targetUs[i] = v;
    }
  } else {
    for(uint8_t i=0;i<N_SERVO;i++){
      uint16_t v = clampServo(i, baseUs[i]);
      currUs[i] = v; targetUs[i] = v;
    }
  }
  prefs.end();
  return okPos || okBase;
}

static void clearNVS(){
  prefs.begin("servos", false);
  prefs.clear();
  prefs.end();
}

static void setOE(bool enabled){
#if USE_OE
  // PCA9685 OE is active-LOW
  digitalWrite(OE_PIN, enabled ? LOW : HIGH);
#else
  (void)enabled;
#endif
}

/** Force Poker neutral at boot: set both current & target to base and write. */
static void forceNeutralAtBoot(){
  portENTER_CRITICAL(&mux);
  for(uint8_t i=0;i<N_SERVO;i++){
    uint16_t v = clampServo(i, baseUs[i]);
    currUs[i] = v;
    targetUs[i] = v;
  }
  portEXIT_CRITICAL(&mux);
  writeAllCurr();  // first write while OE is disabled
}

static void writeAllCurr(){
  for(uint8_t i=0;i<N_SERVO;i++) writeServoUs(i, currUs[i]);
}

static void neutralTargets(){
  portENTER_CRITICAL(&mux);
  for(uint8_t i=0;i<N_SERVO;i++){
    targetUs[i] = clampServo(i, baseUs[i]);
  }
  portEXIT_CRITICAL(&mux);
}

// ── Tasks ────────────────────────────────────────────────────────────────────

/**
 * @brief 50 Hz slew task. Computes next currUs from targetUs under MAX_STEP_US
 *        and pushes I²C writes (outside critical section).
 * @note  Pinned to core 0 to keep Serial (core 1) responsive.
 */
static void TaskSlew(void*){
  const TickType_t dt = pdMS_TO_TICKS(20); // 50 Hz
  uint16_t toWrite[N_SERVO];

  for(;;){
    // Compute next currUs safely (short CS)
    portENTER_CRITICAL(&mux);
    for(uint8_t i=0;i<N_SERVO;i++){
      int32_t diff = (int32_t)targetUs[i] - (int32_t)currUs[i];
      if(diff >  (int32_t)MAX_STEP_US)      currUs[i] += MAX_STEP_US;
      else if(diff < -(int32_t)MAX_STEP_US) currUs[i] -= MAX_STEP_US;
      else                                  currUs[i]  = targetUs[i];
      toWrite[i] = currUs[i];
    }
    portEXIT_CRITICAL(&mux);

    // Perform I2C writes with interrupts enabled
    for(uint8_t i=0;i<N_SERVO;i++) writeServoUs(i, toWrite[i]);

    vTaskDelay(dt);
  }
}

/**
 * @brief Serial RX task. Non-blocking read of Serial, assemble lines, enqueue.
 * @note  Pinned to core 1 (same side as USB Serial in Arduino).
 */
static void TaskSerialRx(void*){
  char buf[256];
  size_t len = 0;

  for(;;){
    while(Serial.available()){
      char c = (char)Serial.read();

      if(c == '\n' || c == '\r'){
        if(len > 0){
          LineMsg msg{};
          size_t n = (len >= sizeof(msg.line)-1) ? sizeof(msg.line)-1 : len;
          memcpy(msg.line, buf, n);
          msg.line[n] = '\0';
          xQueueSend(qLines, &msg, 0);   // best-effort; drop if full
          len = 0;
        }
      } else {
        if(len < sizeof(buf)-1){
          buf[len++] = c;
        } else {
          // Overflow: reset the line to avoid unbounded growth
          len = 0;
        }
      }
    }
    vTaskDelay(pdMS_TO_TICKS(2));
  }
}

/**
 * @brief Command parser task. Pops lines from queue and runs handleLine().
 * @note  Only this task accesses Preferences/NVS and prints protocol replies.
 */
static void TaskCmd(void*){
  LineMsg msg;
  for(;;){
    if(xQueueReceive(qLines, &msg, portMAX_DELAY) == pdTRUE){
      String line = String(msg.line);
      handleLine(line);
    }
  }
}

/**
 * @brief Debounced button task. Press (LOW) → NEUTRAL (failsafe).
 */
static void TaskButton(void*){
  bool prev = true;               // pull-up idle HIGH
  uint32_t lastEdgeMs = millis();
  for(;;){
    bool now = digitalRead(BUTTON_PIN);
    if(now != prev){
      prev = now;
      lastEdgeMs = millis();
    }
    if(!now && (millis() - lastEdgeMs) > 30){
      neutralTargets();           // set targets to base
    }
    vTaskDelay(pdMS_TO_TICKS(5));
  }
}

// ── Command parsing (same functionality) ─────────────────────────────────────
static void handleLine(String line){
  line.trim();
  if(line.length()==0) return;

  // Upper-case command keyword only
  String cmd, rest;
  int sp = line.indexOf(' ');
  if(sp < 0){ cmd = line; rest = ""; }
  else{ cmd = line.substring(0, sp); rest = line.substring(sp+1); rest.trim(); }
  cmd.toUpperCase();

  if(cmd == "PING"){ Serial.println("PONG"); return; }

  if(cmd == "NEUTRAL" || (cmd=="FACE" && rest.equalsIgnoreCase("neutral"))){
    neutralTargets();
    Serial.println("OK NEUTRAL");
    return;
  }

  if(cmd == "POSE"){
    int nextSp = rest.indexOf(' ');
    if(nextSp < 0){ Serial.println("ERR usage: POSE <id> <us>"); return; }
    int id = rest.substring(0, nextSp).toInt();
    uint16_t us = rest.substring(nextSp+1).toInt();
    if(id<0 || id>=N_SERVO){ Serial.println("ERR id"); return; }
    us = clampServo(id, us);
    portENTER_CRITICAL(&mux);
    targetUs[id] = us;
    portEXIT_CRITICAL(&mux);
    Serial.println("OK POSE");
    return;
  }

  if(cmd == "RAW"){
    int nextSp = rest.indexOf(' ');
    if(nextSp < 0){ Serial.println("ERR usage: RAW <id> <us>"); return; }
    int id = rest.substring(0, nextSp).toInt();
    uint16_t us = rest.substring(nextSp+1).toInt();
    if(id<0 || id>=N_SERVO){ Serial.println("ERR id"); return; }
    us = clampServo(id, us);
    portENTER_CRITICAL(&mux);
    currUs[id] = targetUs[id] = us;
    portEXIT_CRITICAL(&mux);
    writeServoUs(id, us);
    Serial.println("OK RAW");
    return;
  }

  if(cmd == "POSES"){
  // Accept exactly 13 (facial+jaw only, ch 0..12) or 16 (full, ch 0..15)
  uint16_t vals[N_SERVO];
  int count = 0, start = 0;

  for(;;){
    int comma = rest.indexOf(',', start);
    String tok = (comma<0) ? rest.substring(start) : rest.substring(start, comma);
    tok.trim();
    if(tok.length()>0){ vals[count++] = (uint16_t)tok.toInt(); }
    if(comma<0) break; else start = comma + 1;
    if(count > N_SERVO) break;
  }

  if(count != 13 && count != 16){
    Serial.println("ERR need 13 or 16 values");
    return;
  }

  portENTER_CRITICAL(&mux);
  // Always set first 13 channels from input
  for(uint8_t i=0;i<13;i++){
    targetUs[i] = clampServo(i, vals[i]);
  }

  if(count == 16){
    // Full frame: also set 13..15 from input
    for(uint8_t i=13;i<N_SERVO;i++){
      targetUs[i] = clampServo(i, vals[i]);
    }
  } else {
    // 13-value legacy: keep neck channels (13..15) unchanged
    for(uint8_t i=13;i<N_SERVO;i++){
      targetUs[i] = clampServo(i, targetUs[i]);
    }
  }
  portEXIT_CRITICAL(&mux);

  Serial.println("OK POSES");
  return;
}

  if(cmd == "SAVE"){
    savePose();
    Serial.println("OK SAVE");
    return;
  }

  if(cmd == "SETBASE"){
  // SETBASE us0,...us12 (13) or us0,...us15 (16)
  uint16_t vals[N_SERVO]; 
  int count=0, start=0;

  for(;;){
    int comma = rest.indexOf(',', start);
    String tok = (comma<0) ? rest.substring(start) : rest.substring(start, comma);
    tok.trim();
    if(tok.length()>0) vals[count++] = (uint16_t)tok.toInt();
    if(comma<0) break; else start = comma+1;
    if(count > N_SERVO) break;
  }

  if(count != 13 && count != 16){
    Serial.println("ERR need 13 or 16 values");
    return;
  }

  // Always set first 13 base values from input
  for(uint8_t i=0;i<13;i++){
    baseUs[i] = clampServo(i, vals[i]);
  }

  if(count == 16){
    // Full frame: also set neck channels from input
    for(uint8_t i=13;i<N_SERVO;i++){
      baseUs[i] = clampServo(i, vals[i]);
    }
  } else {
    // 13-value legacy: keep existing base for neck channels
    for(uint8_t i=13;i<N_SERVO;i++){
      baseUs[i] = clampServo(i, baseUs[i]);
    }
  }

  savePose();
  Serial.println("OK SETBASE");
  return;
}

  if(cmd == "PRINTPOS"){
    printKeyChannels("PRINT");
    printAllChannels("ALL");
    return;
  }

  if(cmd == "CLEARNVS"){
    clearNVS();
    Serial.println("OK CLEARNVS");
    return;
  }

  if(cmd == "LOADPOS"){
    bool had = loadPose();
    writeAllCurr();
    Serial.println(had ? "OK LOADPOS" : "OK LOADPOS (defaults)");
    printKeyChannels("AFTER LOAD");
    return;
  }

  if(cmd == "FACE"){
    // only neutral implemented now
    Serial.println("ERR face not found");
    return;
  }

  Serial.println("ERR unknown cmd");
}

// ── Setup / loop ─────────────────────────────────────────────────────────────
void setup(){
#if USE_OE
  pinMode(OE_PIN, OUTPUT);
  setOE(false);                    // keep outputs DISABLED during init
#endif
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);           // 400 kHz I2C
  pwm.begin();
  pwm.setPWMFreq(SERVO_FREQ);

  Serial.begin(2000000);
  while(!Serial && millis() < 2000) { delay(1); }
  Serial.println();

  // Load saved base/pose (or defaults), then FORCE neutral at boot
  bool hadNVS = loadPose();        // ensures baseUs[] valid; curr/target prepared
  forceNeutralAtBoot();            // write base to hardware while OE disabled
  Serial.println(hadNVS ? "[BOOT] NVS restored" : "[BOOT] using DEFAULT_BASE");
  printKeyChannels("BOOT");
#if USE_OE
  setOE(true);                     // enable outputs AFTER neutral is written
#endif

  // Create command queue
  qLines = xQueueCreate(10, sizeof(LineMsg));

  // Start tasks
  xTaskCreatePinnedToCore(TaskSlew,     "Slew",     4096, nullptr, 2, nullptr, 0); // core 0
  xTaskCreatePinnedToCore(TaskSerialRx, "SerialRx", 4096, nullptr, 3, nullptr, 1); // core 1
  xTaskCreatePinnedToCore(TaskCmd,      "Cmd",      6144, nullptr, 3, nullptr, 1); // core 1
  xTaskCreatePinnedToCore(TaskButton,   "Button",   2048, nullptr, 1, nullptr, 0); // core 0
}

void loop(){
  // Unused. All work is in FreeRTOS tasks.
  vTaskDelay(pdMS_TO_TICKS(1000));
}
