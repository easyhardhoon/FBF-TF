#include "tensorflow/lite/workframe.h"

namespace tflite{

  WorkFrame::WorkFrame(){
    // Creates a new scheduler
    CreateProfilerandScheduler();
  };


  TfLiteStatus WorkFrame::CreateProfilerandScheduler(){
    // NEED DEBUG: use of make_shared here is quite ambigious.
    std::cout << "WorkFrame : Creates TfLite Interpreter, " << 
                   "Scheduler, Modelfactory" << "\n";
    std::shared_ptr<std::queue<tflite::Job*>> new_job_queue = \
                          std::make_shared<std::queue<tflite::Job*>>();
    interpreter = std::make_shared<tflite::Interpreter>();
    std::cout << "Created default interpreter" << "\n";
    scheduler_ = std::make_shared<tflite::Scheduler>(interpreter);
    std::cout << "Created default scheduler" << "\n";
    factory_ = std::make_shared<tflite::ModelFactory>(interpreter);
    std::cout << "Created default modelfactory" << "\n";
    #ifdef DEBUG
      std::cout << "interpreter count :" << interpreter.use_count() << "\n";
    #endif
  };

  TfLiteStatus WorkFrame::CreateAndGiveJob(const char* model){
    if(factory_->CreateProfileModel(model) != kTfLiteOk){
      std::cout << "CreateProfileModel returned ERROR" << "\n";
      return kTfLiteError;
    }
    std::cout << "Factory : succesfully created profilemodel" << "\n";
    scheduler_->NeedReschedule(true);
    return kTfLiteOk;
  };

  TfLiteStatus WorkFrame::AllocateTask(){

  };

  TfLiteStatus WorkFrame::TestInvoke(){
    scheduler_->notify();
    std::cout << "notify \n";
    return kTfLiteOk;
  };

  WorkFrame::~WorkFrame(){
    
  };




} //namespace tflite