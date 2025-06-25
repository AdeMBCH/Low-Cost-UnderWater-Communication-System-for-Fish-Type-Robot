/*
 * iq_transmitter.c
 *
 *  Created on: Jun 3, 2025
 *      Author: adeas
 */



#include "iq_transmitter.h"
#include "stm32f4xx_hal.h"

extern SPI_HandleTypeDef hspi1;
extern TIM_HandleTypeDef htim2;

static IQTransmitter iq_tx;

#define MCP4922_CS_GPIO GPIOA
#define MCP4922_CS_PIN GPIO_PIN_4

static void MCP4922_Select(void) {
    HAL_GPIO_WritePin(MCP4922_CS_GPIO, MCP4922_CS_PIN, GPIO_PIN_RESET);
}

static void MCP4922_Unselect(void) {
    HAL_GPIO_WritePin(MCP4922_CS_GPIO, MCP4922_CS_PIN, GPIO_PIN_SET);
}

static uint16_t MCP4922_Pack(uint8_t channel, uint16_t val12) {
    val12 &= 0x0FFF; // 12 bits
    uint16_t ctrl = (channel ? 0xB000 : 0x3000); // Channel B : 1, A : 0, Gain=1x, buffered
    return ctrl | val12;
}

void IQTransmitter_InitFromBuffer(const int16_t* signal, uint16_t length) {
    iq_tx.buffer = signal;
    iq_tx.length = length;
    iq_tx.index = 0;
    iq_tx.active = 0;
}


void IQTransmitter_Start(void) {
    iq_tx.active = 1;
    iq_tx.index = 0;
    HAL_TIM_Base_Start_IT(&htim2);
}

void IQTransmitter_Stop(void) {
    iq_tx.active = 0;
    HAL_TIM_Base_Stop_IT(&htim2);

    //Forcer la sortie à 0V
    uint16_t spi_word = MCP4922_Pack(0, 0); // canal A, valeur 0
    uint8_t spi_buf[2] = { (spi_word >> 8) & 0xFF, spi_word & 0xFF };

    MCP4922_Select();
    HAL_SPI_Transmit(&hspi1, spi_buf, 2, HAL_MAX_DELAY);
    MCP4922_Unselect();
}
/*
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) {
    if (htim->Instance == TIM2 && iq_tx.active) {
        if (iq_tx.index < iq_tx.length) {
            int16_t sample = iq_tx.buffer[iq_tx.index++];
            uint16_t val = (sample + 2048 > 4095) ? 4095 : (sample + 2048 < 0 ? 0 : sample + 2048);
            uint16_t spi_word = MCP4922_Pack(0, val); // voie A uniquement

            MCP4922_Select();
            HAL_SPI_Transmit(&hspi1, (uint8_t*)&spi_word, sizeof(spi_word), HAL_MAX_DELAY);
            MCP4922_Unselect();
        } else {
            IQTransmitter_Stop();
        }
    }
}*/

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) {
    if (htim->Instance == TIM2 && iq_tx.active) {
        if (iq_tx.index < iq_tx.length) {
        	int16_t sample = iq_tx.buffer[iq_tx.index++];
        	uint16_t val = (sample + 2048 > 4095) ? 4095 : (sample + 2048 < 0 ? 0 : sample + 2048);

            uint16_t spi_word = MCP4922_Pack(0, val); // voie A uniquement

            uint8_t spi_buf[2];
            spi_buf[0] = (spi_word >> 8) & 0xFF;
            spi_buf[1] = spi_word & 0xFF;

            MCP4922_Select();
            HAL_SPI_Transmit(&hspi1, spi_buf, 2, HAL_MAX_DELAY);
            MCP4922_Unselect();
        } else {
            IQTransmitter_Stop();
        }
    }
}

void Enable_TIM2_Interrupt(void) {
    HAL_NVIC_SetPriority(TIM2_IRQn, 1, 0);
    HAL_NVIC_EnableIRQ(TIM2_IRQn);
}

uint8_t IQTransmitter_IsActive(void) {
    return iq_tx.active;
}
