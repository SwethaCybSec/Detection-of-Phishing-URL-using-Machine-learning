#!/usr/bin/env python
# coding: utf-8

# In[138]:


import pandas as pd


# In[139]:


df= pd.read_csv("C://Users//Swetha//OneDrive//Desktop//Tech.csv")


# In[140]:


from urllib.parse import urlparse,urlencode
import re


# In[141]:


def havingIP(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        return 1
    else:
        return 0


# In[142]:


df['havingIP'] = df['url'].apply(lambda x: havingIP(x))


# In[143]:


def haveAtSign(url):
  if "@" in url:
    at = 1    
  else:
    at = 0    
  return at


# In[144]:


df['haveAtSign'] = df['url'].apply(lambda x: haveAtSign(x))


# In[145]:


def getLength(url):
  if len(url) < 54:
    length = 0            
  else:
    length = 1            
  return length


# In[146]:


df['getLength'] = df['url'].apply(lambda x: getLength(x))


# In[147]:


def getDepth(url):
  s = urlparse(url).path.split('/')
  depth = 0
  for j in range(len(s)):
    if len(s[j]) != 0:
      depth = depth+1
  return depth


# In[148]:


df['getDepth'] = df['url'].apply(lambda x: getDepth(x))


# In[149]:


def redirection(url):
  pos = url.rfind('//')
  if pos > 6:
    if pos > 7:
      return 1
    else:
      return 0
  else:
    return 0


# In[150]:


df['redirection'] = df['url'].apply(lambda x: redirection(x))


# In[151]:


def httpsDomain(url):
  domain = urlparse(url).netloc
  if 'https' in domain:
    return 1
  else:
    return 0


# In[152]:


df['httpsDomain'] = df['url'].apply(lambda x: httpsDomain(x))


# In[153]:


shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"                       r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"                       r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"                       r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|"                       r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|"                       r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|"                       r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|"                       r"tr\.im|link\.zip\.net"


# In[154]:


def short_ser(url):
    match=re.search(shortening_services,url)
    if match:
        return 1
    else:
        return 0


# In[155]:


df['short_ser'] = df['url'].apply(lambda x: short_ser(x))


# In[156]:


def prefixSuffix(url):
    if '-' in urlparse(url).netloc:
        return 1            
    else:
        return 0     


# In[157]:


df['prefixSuffix'] = df['url'].apply(lambda x: prefixSuffix(x))


# In[158]:


def misleading_words(url):
    match = re.search('PayPal|secure|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url)
    if match:
        return 1
    else:
        return 0
df['misL_words'] = df['url'].apply(lambda i: misleading_words(i))


# In[159]:


def dot_count(url):
    dot_count = url.count(".")
    if dot_count >= 2:
        return 0
    else:
        return 1


df['dots'] = df['url'].apply(lambda x: dot_count(x))


# In[160]:


def semicolon(url):
    if ';' in url:
        return 1
    else:
        return 0

df['semicolon'] = df['url'].apply(lambda x: semicolon(x))


# In[161]:


def https_or_shttp(url):
    if "https" in url or "shttp" in url:
        return 0
    else:
        return 1

df['https_or_shttp'] = df['url'].apply(lambda x: https_or_shttp(x))


# In[162]:
def shortest_word_length(url):
    words = word_tokenize(url)
    if not words:
        return 0  
    shortest_len = min(len(word) for word in words)
    return shortest_len

tld_list = [".org", ".net", ".gov", ".edu", ".mil", ".int", ".eu", ".coop", ".aero", ".museum", ".jobs", ".mobi", ".cat", ".pro", ".tel", ".travel", ".asia", ".post", ".name", ".guru", ".store", ".app", ".blog", ".design", ".online", ".tech", ".space", ".website", ".store", ".fun", ".site", ".party", ".click", ".link", ".golf", ".club", ".gmbh", ".ltd", ".io", ".ly", ".me", ".to", ".us", ".uk", ".ca", ".au", ".nz", ".ru", ".jp", ".cn", ".kr", ".br", ".za", ".ae", ".in", ".sg", ".ch"]

def tld(url):
    for tld in tld_list:
        if re.search(re.escape(tld), url):
            return 0
    return 1

df['tld'] = df['url'].apply(lambda x: tld(x))


# In[163]:


brand_names = ["google", "apple", "microsoft", "amazon", "facebook"]  # Add more brand names as needed


def brand_and_https(url):
    for brand in brand_names:
        if re.search(brand, url, re.IGNORECASE) and url.startswith("https://"):
            return 0
    return 1

df['brand_and_https'] = df['url'].apply(lambda x: brand_and_https(x))

import nltk
from nltk.tokenize import word_tokenize
def find_longest_word_length(url):
    words = word_tokenize(url)
    longest_word_length = max(len(word) for word in words)
    return longest_word_length


import requests


# In[165]:


from googlesearch import search
def google_index(url):
    site = search(url, 5)
    return 1 if site else 0
df['google_index'] = df['url'].apply(lambda i: google_index(i))


# In[166]:


def generate_punny_code(url):
    
    domain = urlparse(url).netloc

    
    common_tlds = ['com', 'org', 'net', 'io', 'gov', 'edu', 'co', 'io']
    domain_parts = domain.split('.')
    cleaned_domain_parts = [part for part in domain_parts if part not in common_tlds]

   
    punny_code = ''.join(cleaned_domain_parts)

    
    punny_code = re.sub(r'[^a-zA-Z0-9]', '', punny_code).lower()

    punny_code = 1 if len(url) > 63 else 0
    return punny_code


# In[167]:


df['Punny_Code'] = df['url'].apply(generate_punny_code)


# In[ ]:




df


# In[170]:


df.columns


# In[171]:


def about_blank_sfh(url):
    if url.lower().strip() == "about:blank":
        return 1
    else:
        return 0

df['about_blank_sfh'] = df['url'].apply(lambda x: about_blank_sfh(x))


# In[172]:


df['status'] = df['status'].replace({"phishing":1, "legitimate":0})


# In[173]:


X= df[['havingIP', 'haveAtSign', 'getLength', 'getDepth',
       'redirection', 'httpsDomain', 'short_ser', 'prefixSuffix', 'misL_words',
       'dots', 'semicolon', 'https_or_shttp', 'tld', 'brand_and_https',
       'google_index', 'Punny_Code', 'about_blank_sfh']]


# In[174]:


y=df['status']


# In[175]:



from sklearn.model_selection import train_test_split


# In[176]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2,shuffle=True, random_state=5)


# In[177]:


from sklearn.ensemble import RandomForestClassifier


# In[178]:


rf = RandomForestClassifier(n_estimators=100,max_features='sqrt')
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)

import xgboost as xgb
from sklearn.svm import SVC

xgb_c = xgb.XGBClassifier(n_estimators= 100)
xgb_c.fit(X_train,y_train)
y_pred_x = xgb_c.predict(X_test)

svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)
y_pred_svm= svm_model.predict(X_test)



# In[179]:





# In[183]:


def main(url):
    
    status = []
    
    status.append(havingIP(url))
    status.append(haveAtSign(url))
    status.append(getLength(url))
    status.append(getDepth(url))
    status.append(short_ser(url))
    status.append(prefixSuffix(url))
    status.append(misleading_words(url))
    
    status.append(dot_count(url))
    
    status.append(semicolon(url))
    status.append(https_or_shttp(url))
    status.append(tld(url))
    status.append(brand_and_https(url))
    
    status.append(google_index(url))
    status.append(generate_punny_code(url))
    
    status.append(redirection(url))
    status.append(httpsDomain(url))
    status.append(about_blank_sfh(url))
   
        
    return status


# In[186]:


import numpy as np
def get_prediction_from_url(test_url):
    features_test = main(test_url) 
    features_test = np.array(features_test).reshape((1, -1))
    pred = rf.predict(features_test) 


    if int(pred[0]) == 0:
        res = " SAFE"
    else:
        res = " NOT SAFE"

    return res


# In[190]:




# In[ ]:




