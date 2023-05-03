# 2048進化演算法

![DEMO](DEMO.gif)

(4.5倍速播放，當前最高可合成出512, 3188 score)

在看完強化學習的書後決定自己刻一個2048AI來檢驗自己的學習成效

使用`train.py`的`genetic_train()`即可開始訓練

在`play.ipynb`中第一個cell中輸入訓練好的模型的路徑，依序執行每個cell即可實際測試模型效果

---

目前完成基因演算法

進化策略演算法(ES)還在設計中...


## 基因演算法

---

### 訓練邏輯:

我是把神經網路每一層的參數當成基因，每層參數視為生物學上的DNA

#### 模型架構:
```python
params_temp: dict[str, dict[str, Union[torch.Tensor, str, None]]] = {
    'conv1':{
        'w': None, # shape: (out_channels, in_channels, kH, kW)
        'b': None,  # shape: (out_channels)
        'padding': None
    },
    'conv2': {
        'w': None, # shape: (out_channels, in_channels, kH, kW)
        'b': None,  # shape: (out_channels)
        'padding': None
    },
    'layer3': {
        'w': None,
        'b': None
    }
}
```

### 訓練流程:

1. 生成第一代個體
2. 使用多進程繞開python全局鎖，評估個體適應度
3. 依比例隨機選擇出一小群個體
4. 挑出子群體中適應度最好的兩個個體
5. 使這兩個個體做繁殖:
   1. 將兩個體每一層神經網路參數攤平
   2. 對每一層參數做以下行為:
      1. 隨機選擇一個截斷點
      2. 將`A`個體截斷的參數前半部與`B`個體截斷的參數後半部組合，作為子代`A`的參數
      3. 將`B`個體截斷的參數前半部與`A`個體截斷的參數後半部組合，作為子代`B`的參數
   3. 對繁殖後的子代依超參數，隨機變異指定比例的參數
6. 重複步驟`3.`到`5.`，直到子代數量與親代相當
7. 重複步驟`2.`到`6.`，直到進化步長終止

## 環境

---

使用`gym.make('gym2048-v0', disable_env_checker=True)`來建立2048環境

### 環境參數:

* size: 地圖大小
* window_size: 遊戲視窗大小
* interrupt_count: 當代理連續做非法移動達此次樹後終止遊戲
* render_mode: the gym render mode
* is_generate: 除錯用，當為`False`時不會在移動後生成新數字
* game_mode: 當為`'original'`時，會按照原始的2048生成新數字的邏輯生成，否則會隨機生成一個數字`a`並滿足以下條件:
  * `a` 屬於 [sqrt(sqrt(最大值)), sqrt(最大值))
  * log2(`a`) == `int`

### 計分規則:

合成出什麼數字就得幾分，如果同時合成出多個數字，得分為合成出的數字總和

