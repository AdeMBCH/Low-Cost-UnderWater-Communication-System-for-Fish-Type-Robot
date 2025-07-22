#ifndef ASK_MODEM_H
#define ASK_MODEM_H

#include <stdint.h>
#include "stm32f4xx_hal.h"

#define SAMPLE_RATE_HZ 100000
#define SYMBOL_DURATION_US 2000
#define ASK_MAX_BITS 18
#define ASK_SAMPLES_PER_SYMBOL ((SAMPLE_RATE_HZ * SYMBOL_DURATION_US) / 1000000)
#define ASK_RINGBUF_SIZE (ASK_MAX_BITS * ASK_SAMPLES_PER_SYMBOL)

#define THRESH_HIGH 3000.0f   // au-dessus = 1
#define THRESH_LOW  2000.0f   // en-dessous = 0
#define THRESH_GAP  500.0f    // entre 2000 et 3000 = zone morte


typedef struct {
    uint16_t samples_per_symbol;
    float f0; // carrier frequency
    float fs; // sampling frequency
} AskModem;

typedef struct {
    int16_t buf[ASK_RINGBUF_SIZE];
    volatile uint32_t head;
    volatile uint32_t tail;
} AskRingBuffer;

void AskModem_Init(AskModem* modem, uint16_t sample_per_symbol, float f0, float fs);
void AskModem_Modulate(UART_HandleTypeDef* huart, AskModem* modem, const uint8_t* bits, uint16_t bit_len, AskRingBuffer* outbuf, float amplitude);
void AskModem_Demodulate(UART_HandleTypeDef* huart, AskModem* modem, AskRingBuffer* inbuf, uint8_t* bits_out, uint16_t* bit_len);
void AskModem_Modulate_OOK(UART_HandleTypeDef* huart, AskModem* modem, const uint8_t* payload, uint16_t byte_len, AskRingBuffer* outbuf, float amplitude);
//void AskModem_Demodulate_OOK(UART_HandleTypeDef* huart, AskModem* modem, AskRingBuffer* inbuf, uint8_t* bits_out, uint16_t* bit_len);
//void AskModem_Demodulate_OOK(UART_HandleTypeDef* huart, uint16_t* adc_buffer, uint16_t buffer_len, uint16_t samples_per_symbol, uint8_t* bits_out, uint16_t* bit_len);
void AskModem_Demodulate_OOK(UART_HandleTypeDef* huart,uint16_t* adc_buffer,uint16_t buffer_len,uint8_t* bits_out,uint16_t* bit_len);
void AskModem_Demodulate_ByEdges(UART_HandleTypeDef* huart,uint16_t* adc_buffer,uint16_t buffer_len,uint8_t* bits_out,uint16_t* bit_len);

void AskModem_Modulate_DiracTransitions(UART_HandleTypeDef* huart,AskModem* modem,const uint8_t* payload,uint16_t byte_len,AskRingBuffer* outbuf,float amplitude);


void AskRingBuffer_Init(AskRingBuffer* rb);
uint8_t AskRingBuffer_IsFull(const AskRingBuffer* rb);
uint8_t AskRingBuffer_IsEmpty(const AskRingBuffer* rb);
void AskRingBuffer_Put(AskRingBuffer* rb, int16_t value);
int16_t AskRingBuffer_Get(AskRingBuffer* rb);
uint32_t AskRingBuffer_Available(const AskRingBuffer* rb);

#endif
