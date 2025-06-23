/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <string.h>
#include <stdint.h>
#include "uart_protocol.h"
#include "qpsk_modem.h"
#include "CMD.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/*
#define CMD_QPSK_MOD_DEMOD 0x1010
#define CMD_QPSK_RESULT    0x9010
#define CMD_IQ_DATA 0x55AA*/

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;

UART_HandleTypeDef huart2;

QpskModem modem;
QpskRingBuffer tx_ringbuf, rx_ringbuf;

volatile uint8_t bits_to_send[MAX_BITS];
volatile uint16_t num_bits_to_send = 0;
volatile uint16_t bit_idx = 0;
volatile uint8_t transmitting = 0;

/* USER CODE BEGIN PV */

volatile uint8_t qpsk_symbols[QPSK_MAX_SYMBOLS];
volatile uint16_t qpsk_num_symbols = 0;
volatile uint16_t qpsk_symbol_idx = 0;
volatile uint8_t qpsk_transmitting = 0;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_ADC1_Init(void);

/* USER CODE BEGIN PFP */
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

// Callback called when a valid frame is received
/*void OnFrameReceived(UartProtocol* proto, uint16_t cmd, uint16_t len, uint8_t* payload)
{
    // Echo the received frame back to the PC (loopback)
    UartProtocol_SendFrame(&huart2, cmd, len, payload);

}
*/

/*//VRAI QUI MARCHE
void OnFrameReceived(UartProtocol* proto, uint16_t cmd, uint16_t len, uint8_t* payload)
{
    if (cmd == CMD_QPSK_MOD_DEMOD) {

        QpskRingBuffer_Init(&tx_ringbuf);
        QpskRingBuffer_Init(&rx_ringbuf);
        QpskModem_Modulate(&huart2, &modem, payload, len);
        QpskModem_SymbolsToIQ(&modem);
        QpskModem_GenerateSignal(&modem, &tx_ringbuf, 1.0f);

        Qpsk_SimulateReception(&tx_ringbuf, &rx_ringbuf);

        uint8_t data_out[QPSK_MAX_SYMBOLS/4];
        uint16_t len_out = 0;
        QpskModem_Demodulate(&huart2,&modem, &rx_ringbuf, data_out, &len_out);

        //UartProtocol_SendFrame(&huart2, CMD_QPSK_RESULT, len, payload);

       UartProtocol_SendFrame(&huart2, CMD_QPSK_RESULT, len_out, data_out);
    }
}*/

/*
void OnFrameReceived(UartProtocol* proto, uint16_t cmd, uint16_t len, uint8_t* payload)
{
    if (cmd == CMD_QPSK_MOD_DEMOD) {
        QpskRingBuffer_Init(&tx_ringbuf);
        QpskRingBuffer_Init(&rx_ringbuf);
        QpskModem_Modulate(&huart2, &modem, payload, len);
        QpskModem_SymbolsToIQ(&modem);
        QpskModem_GenerateSignal(&modem, &tx_ringbuf, 1.0f);
        Qpsk_SimulateReception(&tx_ringbuf, &rx_ringbuf);
        uint8_t data_out[QPSK_MAX_SYMBOLS/4];
        uint16_t len_out = 0;
        QpskModem_Demodulate(&huart2,&modem, &rx_ringbuf, data_out, &len_out);

        // Conversion QPSK symbols to bits for LED transmission
        num_bits_to_send = 0;
        for (uint16_t i = 0; i < modem.num_symbols; i++) {
            uint8_t symbol = modem.symbols[i]; // 2 bits
            bits_to_send[num_bits_to_send++] = (symbol >> 1) & 0x01; // MSB
            bits_to_send[num_bits_to_send++] = symbol & 0x01;        // LSB
        }
        bit_idx = 0;
        transmitting = 1; // Flag pour démarrer la transmission LED

        UartProtocol_SendFrame(&huart2, CMD_QPSK_RESULT, len_out, data_out);
    }
}
*/


void OnFrameReceived(UartProtocol* proto, uint16_t cmd, uint16_t len, uint8_t* payload)
{
    if (cmd == CMD_QPSK_MOD_DEMOD) {
        QpskRingBuffer_Init(&tx_ringbuf);
        QpskRingBuffer_Init(&rx_ringbuf);
        QpskModem_Modulate(&huart2, &modem, payload, len);
        QpskModem_SymbolsToIQ(&modem);
        QpskModem_GenerateSignal(&modem, &tx_ringbuf, 1.0f);
        Qpsk_SimulateReception(&tx_ringbuf, &rx_ringbuf);

        uint8_t data_out[QPSK_MAX_SYMBOLS/4];
        uint16_t len_out = 0;
        QpskModem_Demodulate(&huart2, &modem, &rx_ringbuf, data_out, &len_out);

        // Prépare les symboles pour la transmission optique
        for (uint16_t i = 0; i < modem.num_symbols; ++i)
            qpsk_symbols[i] = modem.symbols[i];
        qpsk_num_symbols = modem.num_symbols;
        qpsk_symbol_idx = 0;
        qpsk_transmitting = 1; // Drapeau pour démarrer la transmission optique

        UartProtocol_SendFrame(&huart2, CMD_QPSK_RESULT, len_out, data_out);
    }
}

/*
void OnFrameReceived(UartProtocol* proto, uint16_t cmd, uint16_t len, uint8_t* payload)
{
    if (cmd == CMD_QPSK_MOD_DEMOD) {
        // Étape 1 : construire la trame complète UART dans un buffer
        uint8_t full_frame[6 + FRAME_MAX_PAYLOAD];
        int frame_len = UartProtocol_BuildFrame(cmd, len, payload, full_frame);
        if (frame_len < 0) return; // Erreur taille

        // Étape 2 : modulation QPSK
        QpskRingBuffer_Init(&tx_ringbuf);
        QpskRingBuffer_Init(&rx_ringbuf);
        QpskModem_Modulate(&huart2, &modem, full_frame, frame_len);
        QpskModem_SymbolsToIQ(&modem);
        QpskModem_GenerateSignal(&modem, &tx_ringbuf, 1.0f);
        Qpsk_SimulateReception(&tx_ringbuf, &rx_ringbuf);

        // Étape 3 : démodulation (pour test)
        uint8_t data_out[QPSK_MAX_SYMBOLS/4];
        uint16_t len_out = 0;
        QpskModem_Demodulate(&huart2, &modem, &rx_ringbuf, data_out, &len_out);

        // Étape 4 : préparer transmission optique (LED)
        for (uint16_t i = 0; i < modem.num_symbols; ++i)
            qpsk_symbols[i] = modem.symbols[i];

        qpsk_num_symbols = modem.num_symbols;
        qpsk_symbol_idx = 0;
        qpsk_transmitting = 1;

        UartProtocol_SendFrame(&huart2, CMD_QPSK_RESULT, len_out, data_out);
    }
}*/




/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
	UartProtocol proto;
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  MX_ADC1_Init();
  /* USER CODE BEGIN 2 */
  UartProtocol_Init(&proto, OnFrameReceived);

  QpskModem_Init(&modem, 16, 40000.0f, 640000.0f);

	//uint32_t last_toggle = 0;
	//uint32_t period_ms = 200; // 200ms => 2.5Hz
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  uint8_t c;
/*  uint32_t last_bit_tick = 0;
  const uint32_t bit_duration_ms = 20; // Durée d'un bit (ajuste selon ta caméra)

  while (1)
  {
      if (HAL_UART_Receive(&huart2, &c, 1, 10) == HAL_OK) {
          UartProtocol_ParseByte(&proto, c);
      }

      // Transmission LED
      if (transmitting && (HAL_GetTick() - last_bit_tick >= bit_duration_ms)) {
          if (bit_idx < num_bits_to_send) {
              uint8_t bit = bits_to_send[bit_idx++];
              HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, bit ? GPIO_PIN_SET : GPIO_PIN_RESET);
              last_bit_tick = HAL_GetTick();
          } else {
              // Fin de transmission
              HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET);
              transmitting = 0;
          }
      }
  }*/

  uint32_t last_symbol_tick = 0;
  const uint32_t symbol_duration_ms = 100; // Ajuste selon la caméra

  void send_led_preamble(uint32_t symbol_duration_ms) {
      // Ex : 8 symboles alternés
      for (int i = 0; i < 8; ++i) {
          uint8_t left = (i % 2 == 0) ? 1 : 0;
          uint8_t right = (i % 2 == 1) ? 1 : 0;
          HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, left ? GPIO_PIN_SET : GPIO_PIN_RESET);   // LED_A
          HAL_GPIO_WritePin(GPIOA, GPIO_PIN_6, right ? GPIO_PIN_SET : GPIO_PIN_RESET);  // LED_B
          HAL_Delay(symbol_duration_ms); // ou utilise ton timer pour la précision
      }
      // Éteindre les LEDs à la fin du préambule
      HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET);
      HAL_GPIO_WritePin(GPIOA, GPIO_PIN_6, GPIO_PIN_RESET);
  }

  while (1)
  {
      if (HAL_UART_Receive(&huart2, &c, 1, 10) == HAL_OK) {
          UartProtocol_ParseByte(&proto, c);
      }

      // Transmission QPSK optique
      if (qpsk_transmitting && (HAL_GetTick() - last_symbol_tick >= symbol_duration_ms)) {
          if (qpsk_symbol_idx == 0) {
              // Juste avant de commencer la transmission QPSK, envoie le préambule
              send_led_preamble(symbol_duration_ms);
          }

          if (qpsk_symbol_idx < qpsk_num_symbols) {
              uint8_t symbol = qpsk_symbols[qpsk_symbol_idx++];
              uint8_t bit0 = (symbol >> 1) & 0x01; // MSB
              uint8_t bit1 = symbol & 0x01;        // LSB
              HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, bit0 ? GPIO_PIN_SET : GPIO_PIN_RESET); // LED_A
              HAL_GPIO_WritePin(GPIOA, GPIO_PIN_6, bit1 ? GPIO_PIN_SET : GPIO_PIN_RESET); // LED_B
              last_symbol_tick = HAL_GetTick();
          } else {
              HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET);
              HAL_GPIO_WritePin(GPIOA, GPIO_PIN_6, GPIO_PIN_RESET);
              qpsk_transmitting = 0;
          }
      }
  }

/*
  while (1)
  {
      if (HAL_UART_Receive(&huart2, &c, 1, 10) == HAL_OK) {
          UartProtocol_ParseByte(&proto, c);
      }

      // Transmission QPSK optique
      if (qpsk_transmitting && (HAL_GetTick() - last_symbol_tick >= symbol_duration_ms)) {
          if (qpsk_symbol_idx < qpsk_num_symbols) {
              uint8_t symbol = qpsk_symbols[qpsk_symbol_idx++];
              // Mapping QPSK -> LEDs (2 bits)
              uint8_t bit0 = (symbol >> 1) & 0x01; // MSB
              uint8_t bit1 = symbol & 0x01;        // LSB
              HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, bit0 ? GPIO_PIN_SET : GPIO_PIN_RESET); // LED_A
              HAL_GPIO_WritePin(GPIOA, GPIO_PIN_6, bit1 ? GPIO_PIN_SET : GPIO_PIN_RESET); // LED_B
              last_symbol_tick = HAL_GetTick();
          } else {
              // Fin de transmission
              HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET);
              HAL_GPIO_WritePin(GPIOA, GPIO_PIN_6, GPIO_PIN_RESET);
              qpsk_transmitting = 0;
          }
      }
  }*/

  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ADC1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC1_Init(void)
{

  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */

  /** Configure the global features of the ADC (Clock, Resolution, Data Alignment and number of conversion)
  */
  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV2;
  hadc1.Init.Resolution = ADC_RESOLUTION_12B;
  hadc1.Init.ScanConvMode = DISABLE;
  hadc1.Init.ContinuousConvMode = DISABLE;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.NbrOfConversion = 1;
  hadc1.Init.DMAContinuousRequests = DISABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_0;
  sConfig.Rank = 1;
  sConfig.SamplingTime = ADC_SAMPLETIME_3CYCLES;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  /* USER CODE BEGIN MX_GPIO_Init_1 */
	GPIO_InitTypeDef GPIO_InitStruct = {0};
	__HAL_RCC_GPIOA_CLK_ENABLE();
	GPIO_InitStruct.Pin = GPIO_PIN_5;
	GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
	HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);


	GPIO_InitStruct.Pin = GPIO_PIN_6;
	HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOA_CLK_ENABLE();

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
