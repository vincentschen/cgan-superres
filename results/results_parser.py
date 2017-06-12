import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

terminal_output_filename = 'eyeglasses_out' # change to match filename (for filename.txt)
print_every = 400 # 12000/10 # change to set how many batches to print after (1 epoch = 12000 batches)

def readResultsFromFile(fname):
  lines = []
  with open(fname) as f:
    for line in f:
      if 'G_Loss' in line:
        lines.append(line.split())
  return lines

def extractLosses(lines):
  G_Losses = []
  D_Real_Losses = []
  D_Fake_Losses = []
  D_Losses = []
  i = 0
  for line in lines:
    G_Loss = float(line[0].replace('G_Loss[', '').replace('],', ''))
    D_Real_Loss = float(line[1].replace('D_Real_Loss[', '').replace('],', ''))
    D_Fake_Loss = float(line[2].replace('D_Fake_Loss[', '').replace(']', ''))
    D_Loss = D_Real_Loss + D_Fake_Loss
    if i % print_every == 0:
      G_Losses.append(G_Loss)
      D_Real_Losses.append(D_Real_Loss)
      D_Fake_Losses.append(D_Fake_Loss)
      D_Losses.append(D_Loss)
    i += 1
  return G_Losses, D_Real_Losses, D_Fake_Losses, D_Losses


lines = readResultsFromFile(terminal_output_filename + '.txt')

G_Losses, D_Real_Losses, D_Fake_Losses, D_Losses = extractLosses(lines)
print len(G_Losses)
print len(D_Real_Losses)
print len(D_Fake_Losses)
print len(D_Losses)


dis = plt.plot(D_Losses, label='Discriminator Loss')
gen = plt.plot(G_Losses, label='Generator Loss')
plt.legend(['Discriminator Loss', 'Generator Loss'])
plt.ylabel('loss')
# plt.show()
plt.draw()
plt.savefig(terminal_output_filename + '_' + str(print_every) + '.png')

## OTHER POSSIBLE PLOTS:

# plt.plot(D_Losses)
# plt.ylabel('discriminator loss')
# plt.show()

# plt.plot(D_Real_Losses)
# plt.ylabel('discriminator real loss')
# plt.show()

# plt.plot(D_Fake_Losses)
# plt.ylabel('discriminator fake loss')
# plt.show()