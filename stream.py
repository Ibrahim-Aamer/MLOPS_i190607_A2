import requests

def main():
   coin = 'bitcoin'

   response = requests.get("https://api.coincap.io/v2/assets/"+coin)

   print(response)
   print(response.json())


if __name__ == "__main__":
    main()