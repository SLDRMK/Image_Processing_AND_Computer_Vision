import cv2

NUM_OF_GRAYSCALE = 256
ASCLL = 128

class Huffman_Node:
    def __init__(self, value=-1, left=None, right=None):
        self.value = value
        self.frequency = 0
        self.code = ""
        self.left = left
        self.right = right
        self.father = None
        self.last = None
        self.next = None
        
    
    def insert(self, node):
        frequency_flag = node.frequency
        flag = self.next
        while flag.next and flag.frequency < frequency_flag:
            flag = flag.next
        node.next = flag
        flag.last.next = node
        node.last = flag.last
        flag.last = node

    def add_char(self, ch):
        if self.left : self.left.add_char(ch)
        self.code = ch + self.code
        if self.right : self.right.add_char(ch)

class Ordered_List:
    def __init__(self):
        self.head = Huffman_Node()
        self.tail = Huffman_Node()
        self.first = None
        self.length = 0

        self.head.next = self.tail
        self.tail.last = self.head

    def insert(self, node : Huffman_Node):
        self.head.insert(node)
        self.first = self.head.next
        self.length += 1
    
    def remove_first_two(self):
        first : Huffman_Node = self.first
        second : Huffman_Node = first.next

        self.head.next = second.next
        second.next.last = self.head

        self.length -= 2

        return first, second

def read_image(img_src):
    img = cv2.imread(filename=img_src, flags=cv2.IMREAD_GRAYSCALE)
    width, height = img.shape

    nodes = [Huffman_Node(value=value) for value in range(NUM_OF_GRAYSCALE)]

    for row in img:
        for pixel in row:
            nodes[pixel].frequency += 1
    
    list = Ordered_List()

    for node in nodes:
        list.insert(node)

    return img, nodes, list

def build_tree(list : Ordered_List):
    while (list.length > 1):
        first, second = list.remove_first_two()
        f = list.first
        father = Huffman_Node(left=first, right=second)
        father.frequency = first.frequency + second.frequency
        first.add_char('0')
        second.add_char('1')
        first.father = second.father = father
        list.insert(father)
    root = list.first
    return root

def write_compressed_file(dictname, filename, nodes, img):
    dictfile = open(file=dictname, mode='w')
    for node in nodes:
        dictfile.write(node.code + '\n')
    dictfile.close()
    compressed_file = open(file=filename, mode='w')
    temp = 0
    power = 1
    width, height = img.shape
    for row in img:
        for pixel in row:
            for ch in nodes[pixel].code:
                temp += (ch == '1') * power
                power *= 2
                while temp > ASCLL:
                    compressed_file.write(chr(temp % ASCLL))    # write the file as ASCII code
                    temp = int(temp / ASCLL)
                    power = int(power / ASCLL)
    if temp:
        compressed_file.write(chr(temp))
    compressed_file.close()

if __name__ == "__main__":
    # read the image and scan the matrix to calculate distribution of pixel values
    img, nodes, list = read_image("./lena.png")
    f = list.first

    # build the huffman tree
    root = build_tree(list=list)

    length = 0
    width, height = img.shape
    for node in nodes:
        length += len(node.code) * node.frequency
    average_length = length / width / height
    print("average length of pixel for the original file is 8 and for the compressed one is: ", average_length)

    # write the dictionary and compressed file
    write_compressed_file(dictname="./7.1-huffman_dict.txt", filename="./7.1-huffman_compressed_file.txt", nodes=nodes, img=img)