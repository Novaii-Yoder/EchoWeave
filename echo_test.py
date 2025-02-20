from echo_weave import EchoWeave
import os

apikey_file = "../serviceaccount.apikey"

file = open(apikey_file, 'r')
apikey = file.read().strip()
file.close()
os.environ["OPENAI_API_KEY"] = apikey


ew = EchoWeave()
#ew = EchoWeave.load_from_file("echofile.json")

ew.set_api_key(apikey)

ew.add_file("test.txt")
#ew.print_stats()
#ew.remove_reference("test.txt")
#ew.print_stats()
ans = ew.search_chunks_by_query("What horrible things has Pheonix done?")
print(len(ans))
for n in ans:
    print(n[0]["text"])

ew.save_to_file("echofile.json")


