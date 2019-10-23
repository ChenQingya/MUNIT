parser = argparse.ArgumentParser()
parser.add_argument('--compute_CIS', action='store_true', help="whether to compute Conditional Inception Score or not")
opts = parser.parse_args()

if opts.compute_CIS:
    CIS = []

if opts.trainer == 'MUNIT':
    # Start testing
    style_fixed = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
    for i, (images, names) in enumerate(zip(data_loader, image_names)):
        if opts.compute_CIS:
            cur_preds = []
        print(names[1])
        images = Variable(images.cuda(), volatile=True)
        content, _ = encode(images)
        style = style_fixed if opts.synchronized else Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
        for j in range(opts.num_style):
            s = style[j].unsqueeze(0)
            outputs = decode(content, s)
            outputs = (outputs + 1) / 2.
            if opts.compute_IS or opts.compute_CIS:
                pred = F.softmax(inception(inception_up(outputs)), dim=1).cpu().data.numpy()  # get the predicted class distribution
            if opts.compute_IS:
                all_preds.append(pred)
            if opts.compute_CIS:
                cur_preds.append(pred)
            # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
            basename = os.path.basename(names[1])
            path = os.path.join(opts.output_folder+"_%02d"%j,basename)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
        if opts.compute_CIS:
            cur_preds = np.concatenate(cur_preds, 0)
            py = np.sum(cur_preds, axis=0)  # prior is computed from outputs given a specific input
            for j in range(cur_preds.shape[0]):
                pyx = cur_preds[j, :]
                CIS.append(entropy(pyx, py))
        if not opts.output_only:
            # also save input images
            vutils.save_image(images.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)), padding=0, normalize=True)
    if opts.compute_IS:
        all_preds = np.concatenate(all_preds, 0)
        py = np.sum(all_preds, axis=0)  # prior is computed from all outputs
        for j in range(all_preds.shape[0]):
            pyx = all_preds[j, :]
            IS.append(entropy(pyx, py))

    if opts.compute_IS:
        print("Inception Score: {}".format(np.exp(np.mean(IS))))
    if opts.compute_CIS:
        print("conditional Inception Score: {}".format(np.exp(np.mean(CIS))))