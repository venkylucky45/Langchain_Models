from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model = 'openai/gpt-oss-120b')

prompt = PromptTemplate(
    template = 'Explain about this {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

url = 'https://www.flipkart.com/dell-se-series-55-88-cm-22-inch-full-hd-led-backlit-va-panel-contrast-3000-1-tilt-adjustment-1x-hdmi-1xvga-3-years-warranty-tuv-rheinland-3-star-eye-comfort-ultra-thin-bezel-monitor-se2225hm/p/itm928c341bee303?pid=MONHAEETN8ZRFB5K&lid=LSTMONHAEETN8ZRFB5KMIBHBS&marketplace=FLIPKART&store=6bo%2Fg0i%2F9no&srno=b_1_2&otracker=browse&otracker1=hp_rich_navigation_PINNED_neo%2Fmerchandising_NA_NAV_EXPANDABLE_navigationCard_cc_3_L2_view-all&fm=organic&iid=377d4af2-0e4f-4924-b70e-6507c60470af.MONHAEETN8ZRFB5K.SEARCH&ppt=None&ppn=None&ssid=0rodfmz0y80000001767259660765'
loader = WebBaseLoader(url)

chain = prompt | model | parser
docs = loader.load()
# print(docs[0].page_content)
# print(len(docs))
res = chain.invoke({'topic' : docs[0]})
print(res)