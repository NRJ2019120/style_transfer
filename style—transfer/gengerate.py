from torchvision import transforms
from run_code import run_style_transfer
from load_img import load_img, show_img
from torch.autograd import Variable

style_img = load_img('./picture/star.jpg')
# style_img = Variable(style_img).cpu()
style_img = Variable(style_img).cuda()
content_img = load_img('./picture/tower.jpg')
# content_img = Variable(content_img).cpu()
content_img = Variable(content_img).cuda()

input_img = content_img.clone()

out = run_style_transfer(content_img, style_img, input_img, num_epoches=200)
print("style_transfer_img is ...")
show_img(out.cpu())
save_pic = transforms.ToPILImage()(out.cpu().squeeze(0))
save_pic.save('./picture/star_tower.jpg')