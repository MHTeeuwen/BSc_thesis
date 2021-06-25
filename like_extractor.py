
# from bs4 import BeautifulSoup
# import requests
# from requests_html import HTMLSession


# session = HTMLSession()
# response = session.get('https://www.google.com/search?hl=en&ei=coGHXPWEIouUr7wPo9ixoAg&q=%EC%9D%BC%EB%B3%B8%20%E6%A1%9C%E5%B7%9D%E5%B8%82%E7%9C%9F%E5%A3%81%E7%94%BA%E5%8F%A4%E5%9F%8E%20%EB%82%B4%EC%9D%BC%20%EB%82%A0%EC%94%A8&oq=%EC%9D%BC%EB%B3%B8%20%E6%A1%9C%E5%B7%9D%E5%B8%82%E7%9C%9F%E5%A3%81%E7%94%BA%E5%8F%A4%E5%9F%8E%20%EB%82%B4%EC%9D%BC%20%EB%82%A0%EC%94%A8&gs_l=psy-ab.3...232674.234409..234575...0.0..0.251.929.0j6j1......0....1..gws-wiz.......35i39.yu0YE6lnCms')
# soup = BeautifulSoup(response.content, 'html.parser')

# tomorrow_weather = soup.find('span', {'id': 'wob_dc'}).text
# print(tomorrow_weather)






  
# # creating function
# def scrape_info(url):
      
#     # getting the request from url
#     r = requests.get(url)
      
#     # converting the text
#     s = BeautifulSoup(r.text, "html.parser")
      
#     # finding meta info for title
#     title = s.find("span", class_="watch-title").text.replace("\n", "")
      
#     # finding meta info for views
#     views = s.find("div", class_="watch-view-count").text
      
#     # finding meta info for likes
#     likes = s.find("span", class_="like-button-renderer").span.button.text
      
#     # saving this data in dictionary
#     data = {'title':title, 'views':views, 'likes':likes}
      
#     # returning the dictionary
#     return data
  
# # # main function
# # if __name__ == "__main__":
      
# #     # URL of the video
# #     url ="https://www.youtube.com/watch?time_continue=17&v=2wEA8nuThj8"
      
# #     # calling the function
# #     data = scrape_info(url)
      
# #     # printing the dictionary
# #     print(data)



from bs4 import BeautifulSoup
import requests
  
# creating function
def scrape_info(url):
      
    # getting the request from url
    r = requests.get(url)
      
    # converting the text
    s = BeautifulSoup(r.text, "html.parser")
      
    print(s.title)
    # finding meta info for title
    # title = s.find("span", class_="watch-title").text.replace("\n", "")
      
    # # finding meta info for views
    # views = s.find("div", class_="watch-view-count").text
      
    # # finding meta info for likes
    # likes = s.find("span", class_="like-button-renderer").span.button.text
      
    # # saving this data in dictionary
    # data = {'title':title, 'views':views, 'likes':likes}
      
    # returning the dictionary
    return True
  
# main function
if __name__ == "__main__":
      
    # URL of the video
    url ="https://www.youtube.com/watch?time_continue=17&v=2wEA8nuThj8"
      
    # calling the function
    data = scrape_info(url)
      
    # printing the dictionary
    print(data)