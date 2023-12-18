from time import time
import torch 
import torch.nn as nn 

class Attention(nn.Module):
    def __init__(self, input_shape, head):
        super(Attention, self).__init__()
        """
        Initialisiert die Attention-Klasse.
        
        param input_shape: Dimension der Eingabe.
        param head: Anzahl der Attention-Heads.
        """
        self.head = head # Anzahl der attentionheads
        self.input_shape = input_shape # Eingabe-Dimension
        self.head_dims = int(input_shape // head) # Dimension pro Kopf

        # Definition der linearen Transformationen für Query, Key und Value
        self.query = nn.Linear(self.head_dims, self.head_dims)
        self.key = nn.Linear(self.head_dims, self.head_dims)
        self.value = nn.Linear(self.head_dims, self.head_dims)

        # Komplette verbundene Ausgabeschicht
        self.fc = nn.Linear(self.head_dims*head, input_shape)

    def forward(self, query, key, value, mask=None):
        # Batch-Größe und Sequenzlängen nehmen
        batch = query.shape[0]
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]

        # Umformen der Eingaben für Multi-Head Attention
        query, key, value = [
            x.view(batch, -1, self.head, self.head_dims) for x in [query, key, value]
        ]
        
        # linearen Transformationen
        query, key, value = [linear(x) for linear, x in zip([self.query, self.key, self.value], [query, key, value])]

        # Aufmerksamkeitsgewichte
        score = torch.einsum("bqhd,bkhd->bhqk", query, key)
        
        # Masking, falls True
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-1e20"))

        # Softmax fur Normalisierung
        score = torch.softmax(score / (self.head_dims ** 0.5), dim=-1)
        
        # gewichtete Summe
        out = torch.einsum("bhqv,bvhd->bqhd", score, value)
        
        # Umformen und Anwenden der vollständig verbundenen Schicht
        out = out.view(batch, query_len, self.head * self.head_dims)
        out = self.fc(out)
        
        return out # 



class TransformerBlock(nn.Module):
    def __init__(self, input_shape, head, dropout, forward_expansion):
        """
        param input_shape: Größe des Eingabevektors.
        param head: Anzahl der Köpfe in der Multi-Head Attention.
        param dropout: Dropout-Rate zur Vermeidung von Overfitting.
        param forward_expansion: Faktor zur Erweiterung der Dimension im Feedforward-Netzwerk.
        """
        super(TransformerBlock, self).__init__()
        # Initialisiere Multi-Head Attention Layer
        self.attention = Attention(input_shape, head)

        # Initialisiere Feedforward Netzwerk
        self.feed_forward = nn.Sequential(
            nn.Linear(input_shape, input_shape * forward_expansion), # Erweitere Dimension
            nn.GELU(),  # Gelu-Aktivierungsfunktion
            nn.Linear(input_shape * forward_expansion, input_shape)  # Reduziere Dimension zurück
        )

        # Initialisiere Layer-Normalisierungen
        self.layernorm1 = nn.LayerNorm(input_shape)
        self.layernorm2 = nn.LayerNorm(input_shape)
        
        # Initialisiere Dropout-Layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask):
        """
        param query: Query-Matrix.
        param key: Key-Matrix.
        param value: Value-Matrix.
        param mask: Maske zur Steuerung der Aufmerksamkeit.
        return: Ausgabe des Transformer Blocks.
        """
        # Berechne die Aufmerksamkeit und führe Residual-Verbindung durch
        attention = self.attention(query, key, value, mask)
        add = attention + query
        regulazation = self.dropout(self.layernorm1(add))

        # Wende das Feedforward-Netzwerk an und führe eine weitere Residual-Verbindung durch
        forward = self.feed_forward(regulazation)
        out = self.dropout(self.layernorm2(forward + regulazation))

        return out



class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_out,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_len
    ):
        """
        """
        super(Encoder, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_out)
        self.postional_embedding =  nn.Parameter(torch.zeros(1, max_len, embedding_out))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_out,
                    heads,
                    dropout,
                    forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        
    def forward(self, x, mask):
        word_embedding = self.word_embedding(x)
        postional_embedding = self.postional_embedding[:, :x.shape[1], :]
        out = self.dropout(word_embedding + postional_embedding)
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out



class DecoderBlock(nn.Module):
    def __init__(
        self,
        embedding_out,
        head,
        forward_expansion,
        dropout
    ):
        """
        """
        super(DecoderBlock, self).__init__()
        self.attention = Attention(embedding_out, head)
        self.transformer_block = TransformerBlock(
            embedding_out, 
            head, 
            dropout, 
            forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_out)

    def forward(self, query, key, value, src_mask, causal_mask):
        attention = self.attention(query, query, query, causal_mask)
        query = self.dropout(self.norm(attention + query))
        out = self.transformer_block(query, key, value, src_mask)
        return out



class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_out,
        num_layers,
        head,
        forward_expansion,
        dropout,
        max_len
    ):
        super(Decoder, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_out)
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_len, embedding_out))
        self.layers = nn.Sequential(
            *[
            DecoderBlock(
                embedding_out,
                head,
                forward_expansion,
                dropout
            )
            for _ in range(num_layers)
        ]
        )
        self.fc = nn.Linear(embedding_out, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask, casual_mask):
        x = self.dropout(self.word_embedding(x) + self.positional_embedding[:, :x.shape[1], :])
        for layer in self.layers:
            x = layer(
                x, 
                encoder_output, 
                encoder_output, 
                src_mask, 
                casual_mask
            )
        out = self.fc(x)
        return out



class Transformers(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        pad_idx,
        embedding_out,
        num_layers,
        forward_expansion,
        head,
        dropout,
        max_len
    ):
        super(Transformers, self).__init__()
        self.encoder = Encoder(
            input_vocab_size,
            embedding_out,
            num_layers,
            head,
            forward_expansion,
            dropout,
            max_len
        )
        
        self.decoder = Decoder(
            output_vocab_size,
            embedding_out,
            num_layers,
            head,
            forward_expansion,
            dropout,
            max_len
        )
        
        self.pad_idx = pad_idx
        self.apply(self._init_weights)

    #From @HuggingFace
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def pad_mask(self, inputs):
        pad_mask = (inputs != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return pad_mask

    def causal_mask(self, target):
        N, target_len = target.shape
        target_mask = torch.tril(torch.ones((N, target_len, target_len))).unsqueeze(1)
        return target_mask

    def forward(self, inputs, target):
        pad_mask = self.pad_mask(inputs)
        causal_mask = self.causal_mask(target)
        encoder_output = self.encoder(inputs, pad_mask)
        decoder_out = self.decoder(target, encoder_output, pad_mask, causal_mask)
        return decoder_out
        


if __name__ == "__main__":
    # Benutz Character level Tokenizer mit seed
    input_vocab_size = 100
    output_vocab_size = 200

    # Hyper params
    pad_idx = 0 # Damit input zum Transformer gleich lang ist aber nicht mit text
    embedding_out = 512
    num_layers = 6 # Transformer layers
    forward_expansion = 4
    head = 8 # attention Heads
    dropout = 0.1
    max_len = 512 # Kontext lenght

    inputs = torch.randint(0, 100, (32, 200))
    targets = torch.randint(0, 100, (32,100))

    model = Transformers(
        input_vocab_size,
        output_vocab_size,
        pad_idx,
        embedding_out,
        num_layers,
        forward_expansion,
        head,
        dropout,
        max_len
    )

    start = time()
    y = model(inputs, targets)
    print(f'INFERENCE TIME = {time() - start}sek')
    x = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{x} Parameter')
