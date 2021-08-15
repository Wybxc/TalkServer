# TalkServer

这里是我自己训练的一个聊天机器人模型，基于 pytorch 实现。

## 运行

```shell
uvicorn main:app --host 127.0.0.1 --port 6000
```

然后访问 http://127.0.0.1:6000/?msg=你好 (GET 方法)，就可以看到聊天机器人的回复。

响应格式：

```json
{
  "result": "你好。",
  "score": -0.3513331669683759
}
```

| 参数名 | 说明                               |
| ------ | ---------------------------------- |
| result | 聊天机器人的回复                   |
| score  | 回复的分数（基本没有什么参考价值） |

## 在线试用

可以在 https://api.wybxc.cc/talkserver/?msg=你好 体验 TalkServer。
