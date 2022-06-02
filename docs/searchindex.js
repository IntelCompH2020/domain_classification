Search.setIndex({docnames:["dc_base_taskmanager","dc_classifier","dc_custom_model","dc_data_manager","dc_preprocessor","dc_query_manager","dc_task_manager","gui_analyze_keywords_window","gui_constants","gui_get_keywords_window","gui_get_topics_list_window","gui_main_window","gui_messages","gui_output_wrapper","gui_util","gui_worker","gui_worker_signals","index","mn_menu_navigator","usage"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["dc_base_taskmanager.rst","dc_classifier.rst","dc_custom_model.rst","dc_data_manager.rst","dc_preprocessor.rst","dc_query_manager.rst","dc_task_manager.rst","gui_analyze_keywords_window.rst","gui_constants.rst","gui_get_keywords_window.rst","gui_get_topics_list_window.rst","gui_main_window.rst","gui_messages.rst","gui_output_wrapper.rst","gui_util.rst","gui_worker.rst","gui_worker_signals.rst","index.rst","mn_menu_navigator.rst","usage.rst"],objects:{"src.base_taskmanager":[[0,1,1,"","baseTaskManager"]],"src.base_taskmanager.baseTaskManager":[[0,2,1,"","__init__"],[0,3,1,"","__weakref__"],[0,2,1,"","create"],[0,2,1,"","load"],[0,2,1,"","setup"]],"src.data_manager":[[3,1,1,"","DataManager"]],"src.data_manager.DataManager":[[3,2,1,"","__init__"],[3,3,1,"","__weakref__"],[3,2,1,"","get_corpus_list"],[3,2,1,"","get_dataset_list"],[3,2,1,"","get_keywords_list"],[3,2,1,"","get_labelset_list"],[3,2,1,"","get_model_list"],[3,2,1,"","import_labels"],[3,2,1,"","load_corpus"],[3,2,1,"","load_dataset"],[3,2,1,"","load_labels"],[3,2,1,"","load_topics"],[3,2,1,"","reset_labels"],[3,2,1,"","save_dataset"]],"src.domain_classifier":[[1,0,0,"-","classifier"],[2,0,0,"-","custom_model"],[4,0,0,"-","preprocessor"]],"src.domain_classifier.classifier":[[1,1,1,"","CorpusClassifier"]],"src.domain_classifier.classifier.CorpusClassifier":[[1,2,1,"","AL_sample"],[1,2,1,"","__init__"],[1,3,1,"","__weakref__"],[1,2,1,"","annotate"],[1,2,1,"","eval_model"],[1,2,1,"","load_model"],[1,2,1,"","load_model_config"],[1,2,1,"","retrain_model"],[1,2,1,"","train_model"],[1,2,1,"","train_test_split"]],"src.domain_classifier.custom_model":[[2,1,1,"","CustomClassificationHead"],[2,1,1,"","CustomDataset"],[2,1,1,"","CustomEncoderLayer"],[2,1,1,"","CustomModel"]],"src.domain_classifier.custom_model.CustomClassificationHead":[[2,2,1,"","__init__"],[2,2,1,"","forward"]],"src.domain_classifier.custom_model.CustomDataset":[[2,2,1,"","__init__"]],"src.domain_classifier.custom_model.CustomEncoderLayer":[[2,2,1,"","__init__"],[2,2,1,"","forward"]],"src.domain_classifier.custom_model.CustomModel":[[2,2,1,"","__init__"],[2,2,1,"","create_data_loader"],[2,2,1,"","eval_model"],[2,2,1,"","forward"],[2,2,1,"","load_embeddings"],[2,2,1,"","train_model"]],"src.domain_classifier.preprocessor":[[4,1,1,"","CorpusDFProcessor"],[4,1,1,"","CorpusProcessor"]],"src.domain_classifier.preprocessor.CorpusDFProcessor":[[4,2,1,"","__init__"],[4,3,1,"","__weakref__"],[4,2,1,"","compute_keyword_stats"],[4,2,1,"","evaluate_filter"],[4,2,1,"","filter_by_keywords"],[4,2,1,"","filter_by_topics"],[4,2,1,"","get_top_scores"],[4,2,1,"","make_PU_dataset"],[4,2,1,"","make_pos_labels_df"],[4,2,1,"","remove_docs_from_topics"],[4,2,1,"","score_by_keyword_count"],[4,2,1,"","score_by_keywords"],[4,2,1,"","score_by_topics"],[4,2,1,"","score_by_zeroshot"]],"src.domain_classifier.preprocessor.CorpusProcessor":[[4,2,1,"","__init__"],[4,3,1,"","__weakref__"],[4,2,1,"","compute_keyword_stats"],[4,2,1,"","get_top_scores"],[4,2,1,"","performance_metrics"],[4,2,1,"","score_docs_by_keyword_count"],[4,2,1,"","score_docs_by_keywords"],[4,2,1,"","score_docs_by_zeroshot"]],"src.graphical_user_interface":[[7,0,0,"-","analyze_keywords_window"],[8,0,0,"-","constants"],[9,0,0,"-","get_keywords_window"],[10,0,0,"-","get_topics_list_window"],[11,0,0,"-","main_window"],[12,0,0,"-","messages"],[13,0,0,"-","output_wrapper"],[14,0,0,"-","util"],[15,0,0,"-","worker"],[16,0,0,"-","worker_signals"]],"src.graphical_user_interface.analyze_keywords_window":[[7,1,1,"","AnalyzeKeywordsWindow"]],"src.graphical_user_interface.analyze_keywords_window.AnalyzeKeywordsWindow":[[7,2,1,"","__init__"],[7,2,1,"","center"],[7,2,1,"","do_analysis"],[7,2,1,"","initUI"]],"src.graphical_user_interface.constants":[[8,1,1,"","Constants"]],"src.graphical_user_interface.constants.Constants":[[8,3,1,"","__weakref__"]],"src.graphical_user_interface.get_keywords_window":[[9,1,1,"","GetKeywordsWindow"]],"src.graphical_user_interface.get_keywords_window.GetKeywordsWindow":[[9,2,1,"","__init__"],[9,2,1,"","center"],[9,2,1,"","clicked_select_keywords"],[9,2,1,"","init_params"],[9,2,1,"","init_ui"],[9,2,1,"","show_suggested_keywords"],[9,2,1,"","update_params"]],"src.graphical_user_interface.get_topics_list_window":[[10,1,1,"","GetTopicsListWindow"]],"src.graphical_user_interface.get_topics_list_window.GetTopicsListWindow":[[10,2,1,"","__init__"],[10,2,1,"","center"],[10,2,1,"","clicked_get_topic_list"],[10,2,1,"","initUI"],[10,2,1,"","init_params"],[10,2,1,"","show_topics"],[10,2,1,"","synchronize_scrolls"],[10,2,1,"","update_params"]],"src.graphical_user_interface.main_window":[[11,1,1,"","MainWindow"]],"src.graphical_user_interface.main_window.MainWindow":[[11,2,1,"","__init__"],[11,2,1,"","append_text_evaluate"],[11,2,1,"","append_text_retrain_reval"],[11,2,1,"","append_text_train"],[11,2,1,"","center"],[11,2,1,"","clicked_change_predicted_class"],[11,2,1,"","clicked_evaluate_PU_model"],[11,2,1,"","clicked_get_labels"],[11,2,1,"","clicked_get_labels_option"],[11,2,1,"","clicked_give_feedback"],[11,2,1,"","clicked_load_corpus"],[11,2,1,"","clicked_load_labels"],[11,2,1,"","clicked_reevaluate_model"],[11,2,1,"","clicked_reset_labels"],[11,2,1,"","clicked_retrain_model"],[11,2,1,"","clicked_train_PU_model"],[11,2,1,"","clicked_update_ndocs_al"],[11,2,1,"","do_after_evaluate_pu_model"],[11,2,1,"","do_after_give_feedback"],[11,2,1,"","do_after_import_labels"],[11,2,1,"","do_after_load_corpus"],[11,2,1,"","do_after_reevaluate_model"],[11,2,1,"","do_after_retrain_model"],[11,2,1,"","do_after_train_classifier"],[11,2,1,"","execute_evaluate_pu_model"],[11,2,1,"","execute_give_feedback"],[11,2,1,"","execute_import_labels"],[11,2,1,"","execute_load_corpus"],[11,2,1,"","execute_reevaluate_model"],[11,2,1,"","execute_retrain_model"],[11,2,1,"","execute_train_classifier"],[11,2,1,"","init_feedback_elements"],[11,2,1,"","init_ndocs_al"],[11,2,1,"","init_params_train_pu_model"],[11,2,1,"","init_ui"],[11,2,1,"","reset_params_train_pu_model"],[11,2,1,"","show_corpora"],[11,2,1,"","show_labels"],[11,2,1,"","show_sampled_docs_for_labeling"],[11,2,1,"","update_params_train_pu_model"]],"src.graphical_user_interface.messages":[[12,1,1,"","Messages"]],"src.graphical_user_interface.messages.Messages":[[12,3,1,"","__weakref__"]],"src.graphical_user_interface.output_wrapper":[[13,1,1,"","OutputWrapper"]],"src.graphical_user_interface.output_wrapper.OutputWrapper":[[13,2,1,"","__getattr__"],[13,2,1,"","__init__"]],"src.graphical_user_interface.util":[[14,4,1,"","change_background_color_text_edit"],[14,4,1,"","execute_in_thread"],[14,4,1,"","signal_accept"],[14,4,1,"","toggle_menu"]],"src.graphical_user_interface.worker":[[15,1,1,"","Worker"]],"src.graphical_user_interface.worker.Worker":[[15,2,1,"","__init__"],[15,2,1,"","run"]],"src.graphical_user_interface.worker_signals":[[16,1,1,"","WorkerSignals"]],"src.menu_navigator":[[18,0,0,"-","menu_navigator"]],"src.menu_navigator.menu_navigator":[[18,1,1,"","MenuNavigator"]],"src.menu_navigator.menu_navigator.MenuNavigator":[[18,2,1,"","__init__"],[18,3,1,"","__weakref__"],[18,2,1,"","clear"],[18,2,1,"","front_page"],[18,2,1,"","navigate"],[18,2,1,"","query_options"],[18,2,1,"","request_confirmation"]],"src.query_manager":[[5,1,1,"","QueryManager"]],"src.query_manager.QueryManager":[[5,2,1,"","__init__"],[5,3,1,"","__weakref__"],[5,2,1,"","ask_keywords"],[5,2,1,"","ask_label"],[5,2,1,"","ask_label_tag"],[5,2,1,"","ask_labels"],[5,2,1,"","ask_topics"],[5,2,1,"","ask_value"],[5,2,1,"","confirm"]],"src.task_manager":[[6,1,1,"","TaskManager"],[6,1,1,"","TaskManagerCMD"],[6,1,1,"","TaskManagerGUI"]],"src.task_manager.TaskManager":[[6,2,1,"","__init__"],[6,2,1,"","analyze_keywords"],[6,2,1,"","evaluate_PUmodel"],[6,2,1,"","get_feedback"],[6,2,1,"","get_labels_by_keywords"],[6,2,1,"","get_labels_by_topics"],[6,2,1,"","get_labels_by_zeroshot"],[6,2,1,"","get_labels_from_docs"],[6,2,1,"","import_labels"],[6,2,1,"","load"],[6,2,1,"","load_corpus"],[6,2,1,"","load_labels"],[6,2,1,"","reevaluate_model"],[6,2,1,"","reset_labels"],[6,2,1,"","retrain_model"],[6,2,1,"","setup"],[6,2,1,"","train_PUmodel"]],"src.task_manager.TaskManagerCMD":[[6,2,1,"","__init__"],[6,2,1,"","analyze_keywords"],[6,2,1,"","get_labels_by_keywords"],[6,2,1,"","get_labels_by_topics"],[6,2,1,"","get_labels_by_zeroshot"],[6,2,1,"","get_labels_from_docs"],[6,2,1,"","train_PUmodel"]],"src.task_manager.TaskManagerGUI":[[6,2,1,"","get_feedback"],[6,2,1,"","get_labels_by_keywords"],[6,2,1,"","get_labels_by_zeroshot"],[6,2,1,"","get_suggested_keywords"],[6,2,1,"","get_topic_words"],[6,2,1,"","train_PUmodel"]],src:[[0,0,0,"-","base_taskmanager"],[3,0,0,"-","data_manager"],[5,0,0,"-","query_manager"],[6,0,0,"-","task_manager"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function"},terms:{"0":[1,2,4,6,14],"04":18,"05":2,"1":[1,2,3,6,14,18],"100":4,"12":2,"123":4,"19855288":13,"1e":[2,4],"1e100":4,"2":[4,6,18],"2000":6,"2019":18,"2022":2,"3":[1,6],"3072":2,"400":6,"5":1,"58bf8c59b2a3":1,"6":1,"6a7288":14,"768":2,"8":[1,2],"abstract":11,"boolean":[1,3,6],"case":19,"class":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18],"default":[0,1,3,4,5,6,9,10,11,18,19],"do":[0,4,6,19],"float":[1,4,6],"function":[0,1,3,4,5,6,11,14,15],"import":[3,6,11,19],"int":[1,2,4,5,6,14],"jer\u00f3nimo":18,"jes\u00fa":18,"jos\u00e9":2,"new":[0,1,11,19],"public":19,"return":[1,3,4,5,6,16,18],"true":[0,1,3,4,5,6,13],"while":[11,14],A:[1,2,3,4,5,6,18,19],Be:1,For:19,If:[0,1,2,3,4,6,14,18,19],In:[6,13],It:[0,1,3,4,5,6,11,13,15,16],No:1,Not:4,One:18,The:[0,1,2,4,5,6,9,10,11,15,18],Then:19,There:19,To:[0,4,6,19],__getattr__:13,__init__:[0,1,2,3,4,5,6,7,9,10,11,13,15,18],__weakref__:[0,1,3,4,5,8,12,18],_build:19,_get_labelset_list:11,about:[6,17],abov:[4,6],absolut:1,accept:19,accord:[0,4,6,11],account:13,achiev:16,across:1,action:[9,10,11,18],activ:[0,1,6],active_opt:18,adapt:4,add:[3,19],addit:19,after:[4,11,14],al:[1,11],al_sampl:1,algorithm:1,all:[1,3,4,5,6,11,18,19],alreadi:19,an:[0,1,4,5,6,14,18,19],analog:[13,15,16],analysi:7,analyz:17,analyze_keyword:6,analyze_keywords_window:7,analyzekeywordswindow:7,ani:[5,6],annot:[1,11,14],antolin:[1,3,4,6],antonio:2,appear:4,append:14,append_text_evalu:11,append_text_retrain_rev:11,append_text_train:11,apper:4,appli:[1,5],applic:[0,6,7,9,10,11,14,15,16,18,19],appropri:[10,19],approxim:1,ar:[0,1,3,4,6,7,9,10,11,13,14,16,18,19],arena:18,arg:15,argument:[4,14,15],arrai:[4,6],ask:5,ask_keyword:5,ask_label:5,ask_label_tag:5,ask_top:5,ask_valu:5,assign:11,associ:[6,7,9,10,11,14,18],assum:[3,4,19],attach:11,attain:9,attribut:[0,1],author:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18],avail:[1,3,4,10,11,14,16,18],avoid:19,awar:1,back:18,bar:[10,14],bartolom:[6,7,8,9,10,11,12,13,14,15,16],base:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],base_taskmanag:[0,6],basetaskmanag:[0,6],basic:[0,3,4,5,6,19],batch:2,batch_siz:2,been:[9,10,11,13,15,16,19],befor:[1,11,13,18],behavior:[0,6],being:[7,9,10,11,14,16],belong:4,bert:1,between:6,binari:[1,5],blob:14,bodi:4,bool:[0,4,6],border:14,both:[2,14],bound:4,browser:19,bsc:4,build:19,built:19,busi:14,button:[9,10,11],call:1,callback:15,calvo:[6,7,8,9,10,11,12,13,14,15,16],can:[0,5,10,11,19],cannot:11,carri:[6,9,10,11,14],categori:[3,4,6,19],center:[7,9,10,11],chang:[11,14,19],change_background_color_text_edit:14,charg:[10,18],check:[11,19],checkbox:11,chemic:10,cid:[0,1,3,4,5,6,18],class_nam:6,classif:[1,2,4,6],classifi:[2,3,4,6,11],classificationmodel:[1,2],classifier_dropout:2,clean:[0,18],clear:18,click:11,clicked_change_predicted_class:11,clicked_evaluate_pu_model:11,clicked_get_label:11,clicked_get_labels_opt:11,clicked_get_topic_list:10,clicked_give_feedback:11,clicked_load_corpu:11,clicked_load_label:11,clicked_reevaluate_model:11,clicked_reset_label:11,clicked_retrain_model:11,clicked_select_keyword:9,clicked_train_pu_model:11,clicked_update_ndocs_:11,code:[1,4,12,14,19],col:1,col_id:4,collaps:14,collect:18,color:14,columm:1,column:[1,3,4,10],com:[1,13,14,15,16],command:[5,6,19],commit:19,comparison:6,complet:[11,14],compon:[4,6],compos:5,comput:[1,4,6],compute_keyword_stat:4,condit:[4,11],conf:19,config:[0,2,6,9,10,11],config_fnam:[0,6],configreadi:[0,6],configur:[0,1,2,6,7,8,9,10,11,14,19],confirm:[5,18],connect:10,constant:[11,17],contain:[0,1,3,4,5,6,8,11,12,18,19],continu:19,contribut:4,control:[1,9,10,11,14,19],conver_to:5,convers:5,convert_to:5,copi:[1,2],core:6,corpora:11,corpu:[1,3,4,6,7,11],corpus_nam:[3,6],corpusclassifi:[1,4],corpusdfprocessor:4,corpusprocessor:4,correspond:[3,4,6,9,10,11],count:[4,6],creat:[0,1,2,6,13,15,16,18,19],create_data_load:2,cross:19,csv:3,cuda:2,current:11,custom:17,custom_model:2,customclassificationhead:2,customdataset:2,customencoderlay:2,custommodel:2,data:[0,1,2,6,11,16,17,18,19],data_manag:3,datafram:[1,2,3,4,6,11],dataload:2,datamanag:[3,5,6],dataset:[1,2,3,6],dataset_fold:19,db8678:14,defaul:[4,6],defautl:[1,6],defin:[0,1,3,4,5,6,8,12,16,18],delet:3,depend:[0,6,11],descript:[4,10,14],destin:3,detail:19,determinist:1,devic:2,df:2,df_corpu:4,df_dataset:[1,3,4,6],df_eval:2,df_label:[3,4],df_metadata:[4,6],df_out:[1,4],df_stat:4,df_train:2,dict:[4,5,15,18],dictionari:[0,4,5,6,18],differ:19,directli:6,directori:19,displai:[9,11,14],do_after_evaluate_pu_model:11,do_after_give_feedback:11,do_after_import_label:11,do_after_load_corpu:11,do_after_reevaluate_model:11,do_after_retrain_model:11,do_after_train_classifi:11,do_analysi:7,doc:[1,4,6,19],doc_id:4,document:[1,3,4,5,6,7,11,14,19],doe:[1,3],domain:[1,6],domain_classif:3,domain_classifi:[1,2,4],done:19,doubl:11,duplic:13,durat:14,dynam:18,e:[7,9,10,11,19],each:[4,5,6,10,11,14],easiest:1,ecosystem:19,edit:[11,19],element:[4,6,7,9,10,11],embed:[2,3,4,6],emit:13,empti:[3,4,5,11],encod:2,end:14,entri:[0,6],epoch:[1,2],epub:19,equal:[4,5],equival:[1,11],error:[11,16],espinosa:[1,2],eval:2,eval_model:[1,2],eval_pu_classifier_push_button:11,eval_scor:4,evalu:[1,2,4,6,11],evaluate_filt:4,evaluate_pumodel:6,event:14,everi:4,exampl:[1,6],execut:[0,7,9,10,11,14,18],execute_evaluate_pu_model:11,execute_give_feedback:11,execute_import_label:11,execute_in_thread:14,execute_load_corpu:11,execute_reevaluate_model:11,execute_retrain_model:11,execute_train_classifi:11,exist:[0,1,3,6,11,19],exit:18,expand:14,explor:19,extend:6,extens:19,extern:[3,17],extra:6,extrem:1,f_struct:0,facilit:12,factor:[4,6],fals:[3,4,5],feather:3,featur:2,februari:2,feedback:[6,11],file:[0,1,3,5,6,9,10,11,18,19],filenam:3,fill:10,filter_by_keyword:4,filter_by_top:4,finish:[14,16],first:19,fn:15,folder:[0,1,3,4,6,11,19],follow:[1,4,6],followin:0,forc:14,format:[3,4],former:18,forward:2,frequenc:[4,7],from:[0,1,2,3,4,5,6,9,10,11,15,16,18,19],front_pag:18,fulfil:4,function_output:14,futur:19,gallardo:[1,3,4,6],gelu:2,gener:[4,18,19],get:[6,11,17],get_corpus_list:3,get_dataset_list:3,get_feedback:6,get_keywords_list:3,get_keywords_window:9,get_labels_by_keyword:6,get_labels_by_top:6,get_labels_by_zeroshot:6,get_labels_from_doc:6,get_labelset_list:3,get_model_list:3,get_suggested_keyword:6,get_top_scor:4,get_topic_word:6,get_topics_list_window:10,getkeywordswindow:9,gettopicslistwindow:10,github:14,give:11,give_feedback_push_button:11,given:[1,3,4,6,9,10,11,18],go:[9,10,11,14,18,19],gpu:1,graph:7,graphic:[6,17],graphical_user_interfac:[7,8,9,10,11,12,13,14,15,16],gui:[6,7,8,9,10,11,12,14,19],guid:19,ha:[4,11,13,15,16,19],handler:15,have:[4,9,10,11,19],he:10,head:[2,18],help:19,here:19,hidden_act:2,hidden_dropout_prob:2,hidden_s:2,hide:11,high:1,higher:[1,6],highest:1,highlight:19,hold:19,how:19,html:19,http:[1,13,14,15,16],huge:[4,6],i:[4,7,9,10,11],ia:3,ia_keywords_sead_rev_jag:3,icon:[7,9,10,11,14],id:[3,4,10,11],identifi:[3,4],ids_corpu:3,idx:[1,6],import_label:[3,6],improv:6,includ:[1,4,19],incompat:19,index:[4,17,19],indic:[14,16,18],info:[4,17],inform:19,inherit:[6,15],init_feedback_el:11,init_ndocs_:11,init_param:[9,10],init_params_train_pu_model:11,init_ui:[9,11],initi:[1,2,3,4,5,7,9,10,11,15,18],initialis:15,initui:[7,10],input:[1,3,4,5,6],insid:[11,19],instal:19,instanc:[5,13,19],instead:14,instruct:19,intelcomp:4,interact:[5,18,19],interest:19,interfac:[6,17],intermediate_s:2,intern:2,intro:19,introduc:[1,5,11],introduct:19,invok:11,isproject:[0,6],item:11,its:[4,6,7,9,10,11,14],j:[1,3,4,5,6],jesu:0,just:19,k:4,keep:[10,11],kei:[2,18],keyword:[3,4,5,6,10,15,17],kf_stat:4,kw_librari:5,kwarg:15,kwd:6,l6:4,l:[6,7,8,9,10,11,12,13,14,15,16],label:[1,3,4,5,6,11],labelset:6,larg:1,launch:19,layer:2,layer_norm_ep:2,learn:[1,19],least:[1,4],left:14,level:[2,4],librari:1,like:[1,4,19],limit:[4,6],list:[0,1,3,4,5,6,8,9,11,12,15,17,18],ll:19,load:[0,1,2,3,6,11],load_corpu:[3,6],load_dataset:3,load_embed:2,load_label:[3,6],load_model:1,load_model_config:1,load_top:3,local:19,locat:[1,14],log:[4,13],logger:[0,6],loimit:[4,6],look:19,loop:18,low:1,lower:4,magalha:14,main:[0,1,4,5,6,14,15,17,18],main_domain_classifi:19,main_gui:19,main_window:11,maintain:11,mainwindow:[6,11,14],major:12,make:14,make_pos_labels_df:4,make_pu_dataset:4,manag:[11,17,18,19],manual:14,march:18,markdown:17,mask:2,master:14,match:[4,6],matrix:[3,4,6],max_imabal:6,max_imbal:[1,6,11],max_n_doc:11,max_width:14,maximum:[1,4,6,14],maxwidth:14,md:19,me:11,mean:[4,6],menu:[14,17],menu_navig:18,menunavig:18,messag:[17,18],met:11,metadata:[0,4,6],metadata_fnam:[0,6],method:[1,4,5,6,9,10,11,14,18],metric:[1,4],middl:[7,9,10,11],might:[0,1,6],minilm:4,minimum:[4,6,14],mnd:18,mode:17,model:[1,3,4,6,11,17,19],model_nam:4,modeling_roberta:2,modifi:0,modul:[2,3,13,15,16,17],moment:16,more:[16,19],most:19,movement:14,msg:18,much:19,multilevel:18,multipl:1,multithread:[14,15,16],must:[1,4,11,18],myst:19,myst_pars:19,n_doc:[6,11],n_docs_al:11,n_max:[4,6,9,10],n_sampl:1,name:[0,1,3,4,5,6,14,18,19],nativ:19,navig:17,ndarrai:[4,6],necessari:13,need:[5,13,19],neg:[1,6],neural:4,nmax:[1,6,11],nn:2,none:[0,1,3,4,5,6,18],note:[1,4,6],now:19,np:4,num_attention_head:2,num_hidden_lay:2,number:[1,2,4,6,10,11],numer:16,numpi:[4,6],object:[0,1,3,4,5,6,7,8,9,10,11,12,13,14,16,18],occur:6,occurr:6,onc:[11,19],one:[4,11,18],ones:4,onli:[1,3,4,6,11,14],open:[6,7,9,10,11,19],optin:4,option:[0,1,3,4,5,6,14,18,19],order:[6,13],origin:[1,6],other:[1,13,17],otherwis:[1,5],our:19,out:[6,9,10,11,14,19],output:[1,3,4,6,11,17,19],output_wrapp:13,outputwrapp:13,over:[0,1],overrid:13,p_ratio:1,page:[17,19],paht:0,pand:[4,6],panda:[1,3,4],paramet:[0,1,2,3,4,5,6,7,9,10,11,14,15,18,19],parent:[6,13],parti:19,particular:16,pass:[1,14,15],path2dataset:3,path2embed:[3,4],path2label:3,path2menu:18,path2model:3,path2project:[0,6],path2sourc:[0,3,6],path2transform:1,path2zeroshot:[4,6],path:[0,1,3,4,6,11,18,19],path_model:2,pathlib:[0,1,3,4,6,11],paths2data:18,pdf:19,per:[2,4,18],percentag:14,perform:[1,6,7,11],performance_metr:4,pin:19,platform:19,posit:[1,3,4,6,7,9,10,11],possibl:5,potenti:19,practic:[4,6],pre:[4,19],predict:[1,11,14],prefix:1,preprocess:4,preprocessor:[1,17],presenc:[4,7],preserv:[1,6],press:[9,10],pretrain:4,previou:[1,2],primer:19,print:[5,18],printabl:19,probabl:1,process:[0,4,6,16],produc:1,progress:[14,16],progress_bar:14,project:[0,3,4,6,7,9,10,11,19],project_fold:[11,19],projetc:[0,6],proport:[1,6],provid:[0,3,4,5,6,13,14,15,16],pu:[3,11],push:19,py:[14,19],pyqt5:[7,9,10,11,13,15,16],pyqt:[14,15,16],python:19,pythongui:[14,15,16],qcheckbox:11,qdialog:[7,9,10],qmainwindow:11,qobject:[13,16],qprogressbar:14,qradiobutton:11,qrunnabl:15,qstackedwidget:11,qtcore:[13,15,16],qtextedit:[9,11,13,14],qthreadpool:[14,15,16],qtwidget:[7,9,10,11],queri:17,query_manag:5,query_opt:18,querymanag:5,question:13,quick:17,quickstart:19,r:18,random:1,random_st:1,ratio:[1,6],re:1,read:[0,3,5,9,10,11,19],readabl:12,reader:19,rebuild:19,receiv:[11,15],recommend:19,redirect:11,reduc:4,reevalu:11,reevaluate_model:6,refer:[0,1,3,4,5,8,12,18,19],referenc:19,relat:[3,5],relev:4,remain:11,remov:[3,4,6],remove_docs_from_top:4,repositori:[18,19],repres:[7,9,10,11],reproduc:1,request:[6,18],request_confirm:18,requir:[3,4,18],resampl:11,reset:[6,11],reset_label:[3,6],reset_params_train_pu_model:11,resourc:17,respon:18,rest:1,restructuredtext:19,result:[4,16],retrain:11,retrain_model:[1,6],retrain_model_push_button:11,revis:19,roberta:[1,2],robertaclassificationhead:2,robertamodel:2,roof:19,root:19,rout:18,row:[1,10],rst:19,run:[6,15,16,19],runner:15,s:[7,9,10,11,14,15,19],s_min:[4,6,9,10],same:[4,6,11,19],sampl:[1,6,19],sampler:1,save:[1,2,3,19],save_csv:3,save_dataset:3,sbert:4,scipi:4,score:[1,4,6,7],score_by_keyword:4,score_by_keyword_count:4,score_by_top:4,score_by_zeroshot:4,score_docs_by_keyword:4,score_docs_by_keyword_count:4,score_docs_by_zeroshot:4,screen:[7,9,10,11,18],script:18,scriptmodul:2,scroll:10,search:[6,17],second:11,secondari:[11,14],see:19,select:[1,4,5,6,7,9,10,11,14,18,19],selected_doc:6,self:[0,1,3,4,6,11,13],sentenc:2,sequenc:2,seri:[4,8],session:11,set:[0,1,3,4,6,11,14],set_log:[0,6],setup:[0,6,15],sever:1,share:2,shot:[4,6,19],should:[3,4,6,18],show:[6,7,11,14],show_corpora:11,show_label:11,show_sampled_docs_for_label:11,show_suggested_keyword:9,show_top:10,shown:[11,14,18],shuffl:1,signal:[13,15,17],signal_accept:14,signific:4,similar:6,simpl:[1,18],simpletransform:[1,2],simpli:19,singl:[1,4,5],size:[1,4,6],smallest:1,so:[0,4,6,10,11],softwar:19,some:[1,4,5,6,18,19],sort:7,sourc:[0,3,6,11,19],source_fold:11,space:11,spars:4,specif:[0,3,6],specifi:[0,6,9,10,11,14,18],sphinx:17,split:1,src:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18],stackoverflow:13,standard:[5,11],start:[11,15,17],state:[0,2,6],statist:4,stderr:[11,13],stdout:[11,13],step:[6,14],stochast:1,store:[0,1,3],str:[0,1,3,4,5,6,13,18],strictli:[4,6],string:[3,4,5],strongli:19,structur:[0,18],subcorpu:[3,4,9,10,19],subsampl:[1,6],subset:[6,11,18],succesfulli:[0,6],sueiro:[0,1,3,4,5,6,18],suggest:6,suggested_keyword:6,suppli:15,support:[16,19],sure:18,sy:[11,13],synchronize_scrol:10,syntax:19,system:19,t:[4,6],t_i:4,t_out:4,tab:11,tabl:[9,10,11],table_widget_topic_list:10,table_widget_topics_weight:10,tag:[3,5,6],tag_scor:1,take:[1,18],taken:1,target:[1,4,6],target_bio:4,target_col:4,target_en:4,target_t:4,task:[2,11,14,17,18],task_manag:6,taskmanag:[6,7,9,10,11,13,18],taskmanagercmd:6,taskmanagergui:6,te:19,technic:19,templat:19,tensor:2,termin:[17,18],test:[1,6],text:[1,2,4,5,11],text_edit:14,text_edit_results_eval_pu_classifi:11,text_edit_results_reval_retrain:11,text_edit_show_keyword:9,text_logs_train_pu_model:11,than:[1,4,6],thei:[11,19],them:19,thi:[0,1,3,4,5,6,9,10,11,15,19],third:[11,19],thread:[11,14,15,16],three:4,through:[4,5,6,15,18,19],thu:[14,18],time:[4,9,10,11,14],titl:[4,6,7,9,10,11,18],tm:[7,9,10,11,18],toggl:14,toggle_burguer_menu_python_pyside2:14,toggle_menu:14,too:3,top:[9,10],top_prob:1,topic:[3,4,5,6,17],topic_weight:[4,6],topic_word:5,torch:2,towardsdatasci:1,tpc:6,track:10,train:[1,2,4,6,11,19],train_model:[1,2],train_pu_model_push_button:11,train_pumodel:6,train_siz:1,train_test:1,train_test_split:1,transform:[1,2,4],tutori:[14,15,16,19],tw:5,two:[1,3,4,19],txt:3,type:[1,3,4,5,6,18],typic:19,udf:[14,15],ui:[7,9,10,11],ui_funct:14,uncheck:11,undersampl:1,unknown:14,unsort:4,until:[11,19],up:[0,6,15,18],updat:[1,9,10,11,14],update_param:[9,10],update_params_train_pu_model:11,us:[0,1,4,5,6,7,9,10,11,13,15,17,18],use_cuda:1,user:[5,6,9,10,11,14,17,18],util:[2,12,17,19],v2:4,valu:[1,4,5,9,10,11,18],variabl:[0,1,6],vector:4,verbos:4,version:[13,19],vibrant:19,visibl:14,visual:11,vs:[1,6],w_i:4,wa:11,wai:19,walk:19,wanderson:14,want:19,we:19,weak:[0,1,3,4,5,8,12,18],web:19,weight:[4,5,6,10],well:11,were:11,what:19,when:[11,14,19],whenev:13,where:[1,4,5,19],wherev:13,which:[0,6,7,9,10,11,14,19],whole:[1,6],whose:[4,14],widget:[11,13],width:14,window:[5,6,15,17,18,19],within:[9,10,11],word:[4,5,6],work:19,worker:17,worker_sign:16,workersign:16,wrap:[13,15],wrapper:17,write:[3,5,10,19],writer:19,written:13,wt:[4,6,9],www:[14,15,16],x:5,xlm:1,xlnet:1,y:5,yaml:[0,6,9,10,11,18],ye:18,you:[1,18,19],your:19,zero:[4,6,19],zero_opt:18,zero_shot_fold:19,zeroshot:6},titles:["Base Task Manager","Classifier","Custom Model","Data Manager","Preprocessor","Query Manager","Task Manager","Analyze Keywords Window","Constants","Get Keywords Window","Get Topics List Window","Main Window","Messages","Output Wrapper","Util","Worker","Worker Signals","Domain Classifier Docs","Menu Navigator","Getting Started with Domain Classifier"],titleterms:{about:19,analyz:7,base:0,classifi:[1,17,19],constant:8,content:17,custom:2,data:3,doc:17,domain:[17,19],extern:19,get:[9,10,19],graphic:19,indic:17,info:19,interfac:19,keyword:[7,9],list:10,main:11,manag:[0,3,5,6],markdown:19,menu:18,messag:12,mode:19,model:2,navig:18,other:19,output:13,preprocessor:4,queri:5,quick:19,resourc:19,signal:16,sphinx:19,start:19,tabl:17,task:[0,6],termin:19,topic:10,us:19,user:19,util:14,window:[7,9,10,11],worker:[15,16],wrapper:13}})