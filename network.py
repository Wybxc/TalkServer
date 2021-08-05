import torch.nn as nn
import torch.nn.functional as F
import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

class StrPreprocessor:
    """
    字符串预处理模块，包括字符串的张量化，以及字符代码数组的解码。
    注意，这里的字符编码与 Unicode 码不同，是程序内部编码，在每次创建 StrPreprocessor 实例时都不相同。
    """
    def __init__(self, charset):
        """
        :param charset: 一个 set，包含可能处理到的所有字符。
        :type charset: set[str]
        """
        # 字符编码
        charset = charset - {'\n'}
        self.chars = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + list(charset) # 在固定位置添加特殊字符
        self.n_word = len(self.chars) # 字符数量
        self.char2code = {s: i for i, s in enumerate(self.chars)}
        self.char2code['\n'] = 3 # 特殊处理一下换行符，让它指向字符串结束标记
        self.chars[0], self.chars[1], self.chars[2], self.chars[3] = '', '', '', '' # 四个特殊字符不需要显示
        self.PAD, self.UNK, self.SOS, self.EOS = 0, 1, 2, 3

    def str2code(self, s):
        """
        将字符串编码为字符代码数组。

        :type s: str
        :rtype: list[int]
        """
        code = [self.char2code.get(c, self.UNK) for c in s] + [self.EOS]
        return code

    def code2str(self, s):
        """
        将字符代码数组解码为字符串，str2code 的逆过程。

        :type s: list[int]
        :rtype: str
        """
        return ''.join((self.chars[c] for c in s))

    def str2tensor(self, s):
        """
        将字符串编码，并转换为张量。

        :type s: str
        :rtype: torch.LongTensor
        """
        # pylint: disable=not-callable
        return torch.tensor(self.str2code(s), dtype=torch.long, device=device)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_, hidden):
        embedded = self.embedding(input_).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=1000):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_, hidden, encoder_outputs):
        embedded = self.embedding(input_).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
