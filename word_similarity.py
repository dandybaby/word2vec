# 对txt文件进行中文分词
import jieba
import os
from utils import files_processing
from gensim.models import word2vec
import multiprocessing

# 源文件所在目录
source_folder = 'F:\PycharmProjects\word2vec\source'
segment_folder = 'F:\PycharmProjects\word2vec\segment'

# 字词分割，对整个文件内容进行字词分割
def segment_lines(file_list,segment_out_dir,stopwords=[]):
    for i,file in enumerate(file_list):
        segment_out_name=os.path.join(segment_out_dir,'segment_{}.txt'.format(i))
        with open(file, 'rb') as f:
            document = f.read()
            document_cut = jieba.cut(document)
            sentence_segment=[]
            for word in document_cut:
                if word not in stopwords:
                    sentence_segment.append(word)
            result = ' '.join(sentence_segment)
            result = result.encode('utf-8')
            with open(segment_out_name, 'wb') as f2:
                f2.write(result)

# 对source中的txt文件进行分词，输出到segment目录中
file_list=files_processing.get_files_list(source_folder, postfix='*.txt')
segment_lines(file_list, segment_folder)
# 先运行 word_seg进行中文分词，然后再进行word_similarity计算
# 将Word转换成Vec，然后计算相似度


# 如果目录中有多个文件，可以使用PathLineSentences
segment_folder = 'F:\PycharmProjects\word2vec\segment'
sentences = word2vec.PathLineSentences(segment_folder)

# 设置模型参数，进行训练
model = word2vec.Word2Vec(sentences, size=128, window=5, iter=1000, min_count=10, workers=multiprocessing.cpu_count())
model.save('./word2Vec.model')
print('Nearest 曹操:', model.wv.most_similar('曹操'))
print('曹操+刘备-张飞=', model.wv.most_similar(positive=['曹操', '刘备'], negative=['张飞']))

