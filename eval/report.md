# Evaluation report

Run: 2026-11-20T18:46:58+00:00 · model `llama3.2` · 30 questions · APIs mocked

## Scores

| Metric | Passed | Score |
|---|---|---|
| tools | 30/30 | 100% |
| answer | 30/30 | 100% |
| refusal | 30/30 | 100% |
| memory | 2/2 | 100% |
| **all checks passed** | **30/30** | **100%** |

## By category

| Category | All checks passed |
|---|---|
| docs | 10/10 |
| currency | 4/4 |
| weather | 4/4 |
| mixed | 4/4 |
| smalltalk | 2/2 |
| refusal | 4/4 |
| memory | 2/2 |

## Every question

| ID | Result | Tools used | Answer |
|---|---|---|---|
| docs-01 | ✅ | search_documents | You get 20 days of paid annual leave per year. |
| docs-02 | ✅ | search_documents | Full-time employees receive 10 days of paid sick leave per year. |
| docs-03 | ✅ | search_documents | The hotel limit per night for domestic trips is $150. |
| docs-04 | ✅ | search_documents | The meal allowance per day on international trips is $75. |
| docs-05 | ✅ | search_documents | According to the company policy, your password must be at least 12 characters long and include a number and a symbol. |
| docs-06 | ✅ | search_documents | Yes, you need to use the VPN when working from a cafe. According to the company policy, the VPN is required when working outside the office, |
| docs-07 | ✅ | search_documents | The monthly gym allowance is 40 US dollars. |
| docs-08 | ✅ | search_documents | You can work from home up to 2 days per week, with manager approval. |
| docs-09 | ✅ | search_documents | The maximum size of an email attachment is 25 MB. |
| docs-10 | ✅ | search_documents | The salary range for a Senior Engineer at Acme Corp is between $85,000 and $115,000 per year. |
| fx-01 | ✅ | get_exchange_rate | 100 USD is equivalent to 90 EUR. |
| fx-02 | ✅ | get_exchange_rate | You have been converted to 42.5 GBP. |
| fx-03 | ✅ | get_exchange_rate | The current exchange rate from USD to JPY is 1 USD = 150 JPY. |
| fx-04 | ✅ | get_exchange_rate | Sorry, I couldn't get that: Unknown currency code 'USD' or 'XYZ'. Use codes like USD, EUR, JPY. |
| wx-01 | ✅ | get_weather | The current weather in Paris is light rain with a temperature of 8.0°C (feels like 5.0°C) and a wind speed of 10.0 km/h. |
| wx-02 | ✅ | get_weather | It is currently -2.0°C (feels like -6.0°C) in Berlin, Germany, with light snow and a wind speed of 10.0 km/h. |
| wx-03 | ✅ | get_weather | It is currently clear in Tokyo, with a temperature of 12.0°C (feels like 10.0°C) and a gentle breeze of 10.0 km/h. |
| wx-04 | ✅ | get_weather | Sorry, I couldn't get that: City 'Atlantisxyz' not found. Check the spelling, e.g. 'Tokyo' or 'Paris'. |
| mix-01 | ✅ | search_documents, get_exchange_rate, get_weather, get_exchange_rate | The hotel limit in Tokyo is 33,000 yen per night. The current weather in Tokyo is clear with a temperature of 12°C (feels like 10°C). |
| mix-02 | ✅ | search_documents, get_exchange_rate, get_exchange_rate | The meal allowance per day on an international trip is €67.50. |
| mix-03 | ✅ | search_documents, get_exchange_rate | The hotel limit per night for international trips is 176 GBP. |
| mix-04 | ✅ | get_weather, get_exchange_rate | The current weather in Paris is Light rain, 8.0°C (feels like 5.0°C), wind 10.0 km/h. 200 USD is equivalent to 180 EUR. |
| chat-01 | ✅ | — | Hello! How can I assist you today? |
| chat-02 | ✅ | — | You're welcome! How can I assist you today? |
| ref-01 | ✅ | search_documents | I don't know based on the company documents. |
| ref-02 | ✅ | search_documents | I don't know based on the company documents. |
| ref-03 | ✅ | search_documents | I don't know based on the company documents. |
| ref-04 | ✅ | search_documents | I don't know based on the company documents. |
| mem-01 | ✅ | save_memory | I've saved that for future reference. How can I assist you further? |
| mem-02 | ✅ | — | You're a vegetarian, and we've had a previous conversation. |
