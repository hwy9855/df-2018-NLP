CCF-BDCI 2018 汽车行业用户观点主题及情感识别
===
A榜分数0.64088400 排名42/656

B榜分数0.63813990 排名85/672

---
最高分获得步骤为：
```commandline
python3 main.py
python3 baseline.py
```
其中需要调整baseline.py中的输入csv路径为main.py中的输出csv路径。

main.py中含有多个版本的方法：
```python
run()
run_base()
run_base_bdc()
run_boost()
```
可以通过在__main__中调整参数来使用不同的模型。