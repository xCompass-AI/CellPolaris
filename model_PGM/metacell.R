library(proxy)

file_names= list.files('.',pattern ='\\.csv$',full.names = TRUE )
df_sum=read.table('E:/labdata/fpkm计算/mm10_gene_length.txt')

for (file in file_names){
  celltype <- sub(".*/(.*?)\\.csv", "\\1", file)
  scRNA=read.csv(file,header = 1,row.names = 1)
  scRNA=data.frame(t(scRNA))
  dist_matrix <- proxy::dist(scRNA)
 
  dist_matrix <- as.matrix(dist_matrix)

  # 创建一个空的矩阵来存储每50个距离最近的细胞的表达值
  bulk_expression <- matrix(0, nrow = nrow(scRNA), ncol = ncol(scRNA))
  for (i in 1:(nrow(scRNA))) {
    nearest_cells <- order(dist_matrix[i, ])[1:10]  # 排除自身，选择最近的50个细胞
    bulk_expression[i, ] <- apply(scRNA[nearest_cells, ], 2, sum)
  }
  
  # scRNA=data.frame(t(scRNA))
  # n_cells_per_col <- ceiling(dim(scRNA)[2]/10)
  # n_cols <- 100
  # n_genes <- nrow(scRNA)
  # n = ncol(scRNA)
  # df2 = data.frame(matrix(nrow = n_genes, ncol = n_cols))
  df1=data.frame(t(bulk_expression))
  
  # 随机选择50列
  selected_cols <- sample(colnames(df1), 50,replace = TRUE)
  # 创建子数据框
  df2 <- df1[, selected_cols]
  # for (i in 1:n_cols) {
  #   selected_cells <- sample(1:n, n_cells_per_col, replace = FALSE)
  #   # 抽取 n_cells_per_col 个单细胞，不允许重复抽取
  #   # samPLe() 函数可以实现从指定的样本中，按照指定的方式随机抽取特定数量的样本
  #   
  #   df2[,i] <- apply(scRNA[,selected_cells], 1, sum)
  #   # 从 df2 中选出 selected_cells 中对应的列，按行求和，赋值给 df2 中的第 i 列
  #   # apply() 函数可以在数据的行或列上应用指定的函数，实现聚合、计算等功能
  # }
  rownames(df2)=colnames(scRNA)
  df2 = merge(df2,df_sum,by.x = "row.names", by.y = "V1", all.x = FALSE)
  
  for (i in 2:(ncol(df2)-1)){
    N=sum(df2[,i])
    df2[,i]=exp( log(df2[,i]) + log(1e9) - log(df2[,ncol(df2)]) - log(N))
    
  }
  df2=na.omit(df2)
  write.csv(df2[,1:(ncol(df2)-1)],paste(celltype,'sample.csv',sep = '_'),
            row.names = FALSE,quote = FALSE,col.names = FALSE)
  
  
}

