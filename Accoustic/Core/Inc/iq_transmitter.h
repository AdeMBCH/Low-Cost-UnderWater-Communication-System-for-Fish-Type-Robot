/*
 * iq_transmitter.h
 *
 *  Created on: Jun 3, 2025
 *      Author: adeas
 */

#ifndef INC_IQ_TRANSMITTER_H_
#define INC_IQ_TRANSMITTER_H_

#include "stm32f4xx_hal.h"
#include "qpsk_modem.h"

typedef struct {
	const int16_t *buffer;
    //uint16_t *buffer;
    uint32_t length;
    volatile uint32_t index;
    volatile uint8_t active;
} IQTransmitter;


void IQTransmitter_InitFromBuffer(const int16_t* signal, uint16_t length);
void IQTransmitter_Start(void);
void IQTransmitter_Stop(void);

void Enable_TIM2_Interrupt(void);

uint8_t IQTransmitter_IsActive(void);


#endif /* INC_IQ_TRANSMITTER_H_ */
