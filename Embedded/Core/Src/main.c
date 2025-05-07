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
#include "stm32f4xx_hal_conf.h"
#include "stm32f4xx_it.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
#define FRAME_MAX_PAYLOAD 256
uint8_t frame_payload[FRAME_MAX_PAYLOAD];
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */
void ProcessFrame(uint16_t cmd, uint16_t len, uint8_t* payload);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
typedef enum {
    WAIT_SOF,
    READ_CMD_MSB,
    READ_CMD_LSB,
    READ_LEN_MSB,
    READ_LEN_LSB,
    READ_PAYLOAD,
    READ_CHECKSUM
} RX_State;

RX_State rx_state = WAIT_SOF;
uint16_t rx_cmd = 0;
uint16_t rx_len = 0;
uint16_t rx_payload_idx = 0;
uint8_t rx_checksum = 0;
uint8_t rx_calc_checksum = 0;

void ResetRxState() {
    rx_state = WAIT_SOF;
    rx_cmd = 0;
    rx_len = 0;
    rx_payload_idx = 0;
    rx_checksum = 0;
    rx_calc_checksum = 0;
}

uint8_t CalcChecksum(uint16_t cmd, uint16_t len, uint8_t* payload) {
    uint8_t cs = 0;
    cs ^= 0xFE;
    cs ^= (cmd >> 8) & 0xFF;
    cs ^= (cmd >> 0) & 0xFF;
    cs ^= (len >> 8) & 0xFF;
    cs ^= (len >> 0) & 0xFF;
    for (uint16_t i = 0; i < len; i++) {
        cs ^= payload[i];
    }
    return cs;
}

void UART_RxHandler(uint8_t c) {
    switch(rx_state) {
        case WAIT_SOF:
            if (c == 0xFE) {
                rx_state = READ_CMD_MSB;
                rx_calc_checksum = 0xFE;
            }
            break;
        case READ_CMD_MSB:
            rx_cmd = ((uint16_t)c) << 8;
            rx_calc_checksum ^= c;
            rx_state = READ_CMD_LSB;
            break;
        case READ_CMD_LSB:
            rx_cmd |= c;
            rx_calc_checksum ^= c;
            rx_state = READ_LEN_MSB;
            break;
        case READ_LEN_MSB:
            rx_len = ((uint16_t)c) << 8;
            rx_calc_checksum ^= c;
            rx_state = READ_LEN_LSB;
            break;
        case READ_LEN_LSB:
            rx_len |= c;
            rx_calc_checksum ^= c;
            if (rx_len > FRAME_MAX_PAYLOAD) {
                // Too big, reset state
                ResetRxState();
            } else if (rx_len == 0) {
                rx_state = READ_CHECKSUM;
            } else {
                rx_payload_idx = 0;
                rx_state = READ_PAYLOAD;
            }
            break;
        case READ_PAYLOAD:
            frame_payload[rx_payload_idx++] = c;
            rx_calc_checksum ^= c;
            if (rx_payload_idx >= rx_len) {
                rx_state = READ_CHECKSUM;
            }
            break;
        case READ_CHECKSUM:
            rx_checksum = c;
            if (rx_checksum == rx_calc_checksum) {
                // Valid frame
                ProcessFrame(rx_cmd, rx_len, frame_payload);
            }
            // Regardless, reset state
            ResetRxState();
            break;
        default:
            ResetRxState();
            break;
    }
}

// Send a protocol frame (echo)
void SendFrame(uint16_t cmd, uint16_t len, uint8_t* payload) {
    uint8_t tx_buf[6 + FRAME_MAX_PAYLOAD];
    int pos = 0;
    tx_buf[pos++] = 0xFE;
    tx_buf[pos++] = (cmd >> 8) & 0xFF;
    tx_buf[pos++] = (cmd >> 0) & 0xFF;
    tx_buf[pos++] = (len >> 8) & 0xFF;
    tx_buf[pos++] = (len >> 0) & 0xFF;
    for (int i = 0; i < len; i++) {
        tx_buf[pos++] = payload[i];
    }
    uint8_t cs = CalcChecksum(cmd, len, payload);
    tx_buf[pos++] = cs;
    HAL_UART_Transmit(&huart2, tx_buf, pos, 100);
}

// Called when a valid frame is received
void ProcessFrame(uint16_t cmd, uint16_t len, uint8_t* payload) {
    // Echo back the same frame (loopback)
    SendFrame(cmd, len, payload);
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

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
  /* USER CODE BEGIN 2 */
  ResetRxState();
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  uint8_t c;
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	  if (HAL_UART_Receive(&huart2, &c, 1, 10) == HAL_OK) {
		  UART_RxHandler(c); // This will parse the protocol and echo the whole frame
	    }
    }
    // Optionally add a small delay or other tasks
  }
  /* USER CODE END 3 */

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
