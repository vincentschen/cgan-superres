FULL_ATTRIBUTE_LIST = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
NUM_IMGS = 202599
ATTRIBUTE_FILENAME = './datasets/celebA/list_attr_celeba.txt'

class Attributes:
	def __init__(self, inputfile=ATTRIBUTE_FILENAME):
		self.attributeMap = self.readAttributesFromFile(inputfile) # {'000001.jpg': {'5_o_Clock_Shadow': 1, ...}, ...}

	def readAttributesFromFile(self, fname):
		attributeMap = {}
		with open(fname, 'r') as f:
			lineNum = 0
			for line in f:
				lineNum += 1
				if lineNum < 3: # skip the first 2 lines
					continue

				imgAttributes = {}

				line = line.split()
				attributeMap[line[0]] = imgAttributes
				line = line[1:]
				for i in range(len(FULL_ATTRIBUTE_LIST)):
					imgAttributes[FULL_ATTRIBUTE_LIST[i]] = int(line[i])
		return attributeMap

attr = Attributes().attributeMap
