from PIL import Image, ImageDraw, ImageFont
import csv
"""
csv 형식 : [PredictionString, image_id]

PredictionString = (label, score, xmin, ymin, xmax, ymax) 가 공백을 구분자로 구분없이 들어간다. 
ex. bbox가 2개있는 이미지의 PredictionString 는 label1 score1 xmin1 ymin1 xmax1 ymax1 label2 score2 xmin2 ymin2 xmax2 ymax2 와 같다.

"""
colorlist = [(0,0,0),(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,69,0),(0,255,255),(255,0,255),(75,0,130),(128,128,128)]
textlist = ["General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]
csv_path = '/opt/ml/example.csv'
font_path = '/opt/ml/Chalkboard.ttc'
result_path = '/opt/ml/bbound_test_infer_text'
f = open(csv_path,'r')
rdr = csv.reader(f)
cnt = 0
infolist = []
for line in rdr:
    if cnt!=0:
        infolist.append(line)
    cnt+=1
f.close()

for i in infolist:
    bboxinfo , imagepath = i[0], i[1]
    img = Image.open('/opt/ml/dataset/'+imagepath).convert('RGB')
    draw = ImageDraw.Draw(img)
    bboxlist = bboxinfo.split()
    num_of_bbox = len(bboxlist)//6 #bbox의 개수
    for j in range(num_of_bbox):
        colorindex = int(bboxlist[6*j])
        minx, miny, maxx, maxy = float(bboxlist[6*j+ 2]),float(bboxlist[6*j+ 3]),float(bboxlist[6*j+ 4]),float(bboxlist[6*j+5])
        text_pos = (minx+5,miny-40)
        font = ImageFont.truetype(font_path , 30)
        draw.rectangle((minx,miny,maxx,maxy), outline=colorlist[colorindex], width = 5)
        draw.text(text_pos, textlist[colorindex],colorlist[colorindex],font=font) 
    # img.show() -> .py 파일일때는 서버에서 사진 확인이 불가능할 것이다. 확인하고 싶으면 ipynb에서 실행해보자.
    img.save(result_path+'/'+imagepath[5:9]+'.png', 'png')